import torch
import triton
import triton.language as tl



@triton.jit
def compute_seg_indptr_triton_kernel(reorder_topk_ids, seg_indptr, num_toks):
    expert = tl.program_id(0)
    low = 0
    high = num_toks - 1
    target_location = -1
    while low <= high:
        mid = (low + high) // 2

        if tl.load(reorder_topk_ids + mid) > expert:
            high = mid - 1
        else:
            low = mid + 1
            target_location = mid
    tl.store(seg_indptr + expert + 1, target_location + 1)

@triton.jit
def compute_src2dst_triton_kernel(
    reorder_ids, src2dst, num_toks, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    dst_id = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = dst_id < num_toks
    src_id = tl.load(reorder_ids + dst_id, mask=mask)
    tl.store(src2dst + src_id, dst_id, mask=mask)


def run_moe_ep_preproess(topk_ids: torch.Tensor, num_experts: int):
    """
    Args : topk_ids: torch.Tensor, shape (num_tokens, topk)
           num_experts: int (256)

    Returns : reorder_topk_ids: torch.Tensor, shape (num_tokens * topk)
            : src2dst: torch.Tensor, shape (num_tokens * topk)
    """
    reorder_topk_ids, reorder_ids = torch.sort(topk_ids.view(-1), stable=True)
    # print(reorder_topk_ids) # [0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 5, 5, 6, 7]
    # print(reorder_ids)      # [0, 4, 8, 12, 16, 6, 11, 13, 1, 10, 14, 3, 9, 15, 19, 18, 2, 17, 7, 5]
    seg_indptr = torch.zeros(num_experts + 1, device=topk_ids.device, dtype=torch.int64) # shape: (num_experts + 1)
    src2dst = torch.empty(topk_ids.numel(), device=topk_ids.device, dtype=torch.int32) # shape: (num_tokens * topk)

    compute_seg_indptr_triton_kernel[(num_experts,)](
        reorder_topk_ids, seg_indptr, topk_ids.numel()
    )
    # print(seg_indptr) # [ 0,  5,  8, 11, 15, 16, 18, 19, 20]

    BLOCK_SIZE = 512
    grid = (triton.cdiv(topk_ids.numel(), BLOCK_SIZE),)
    compute_src2dst_triton_kernel[grid](
        reorder_ids, src2dst, topk_ids.numel(), BLOCK_SIZE
    )
    print(src2dst) # [ 0, 8, 16, 11, 1, 19, 5, 18, 2, 12, 9, 6, 3, 7, 10, 13, 4, 17, 15, 14 ]
    return reorder_topk_ids, src2dst, seg_indptr



@triton.jit
def pre_reorder_triton_kernel(
    input_ptr,
    gateup_input_ptr,
    src2dst_ptr,
    topk_ids_ptr,
    a1_scales_ptr,
    start_expert_id,
    end_expert_id,
    topk,
    hidden_size,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Args : input_ptr: torch.Tensor, shape (num_tokens, hidden_size)
           gateup_input_ptr: torch.Tensor, shape (batch_size, num_experts, hidden_size)
           src2dst_ptr: torch.Tensor, shape (batch_size, num_tokens * topk)
           topk_ids_ptr: torch.Tensor, shape (batch_size, num_tokens * topk)
           a1_scales_ptr: torch.Tensor, shape (num_experts)
           start_expert_id : rank of the first expert
           end_expert_id : rank of the last expert        
    """
    OutDtype = gateup_input_ptr.dtype.element_ty

    src_idx = tl.program_id(0) # the id of the token
    src2dst_ptr = src2dst_ptr + src_idx * topk # move the pointer to the current token
    topk_ids_ptr = topk_ids_ptr + src_idx * topk # 

    src_ptr = input_ptr + src_idx * hidden_size
    for idx in range(topk):
        expert_id = tl.load(topk_ids_ptr + idx)
        if expert_id >= start_expert_id and expert_id <= end_expert_id:
            if a1_scales_ptr is not None:
                scale = 1.0 / tl.load(a1_scales_ptr + expert_id - start_expert_id)
            else:
                scale = 1.0

            dst_idx = tl.load(src2dst_ptr + idx)
            dst_ptr = gateup_input_ptr + dst_idx * hidden_size
            for start_offset in tl.range(0, hidden_size, BLOCK_SIZE):
                offset = start_offset + tl.arange(0, BLOCK_SIZE)
                mask = offset < hidden_size
                in_data = tl.load(src_ptr + offset, mask=mask).to(tl.float32)
                out_data = (in_data * scale).to(OutDtype)
                tl.store(dst_ptr + offset, out_data, mask=mask)
                

if __name__ == "__main__":
    topk_ids = torch.tensor([[0, 2, 5, 3],
                             [0, 7, 1, 6],
                             [0, 3, 2, 1],
                             [0, 1, 2, 3],
                             [0, 5, 4, 3]], device="cuda")
    # print(topk_ids.shape)
    num_experts = 8
    reorder_topk_ids, src2dst, seg_indptr = run_moe_ep_preproess(topk_ids, num_experts)

    
    
