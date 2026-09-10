import torch
from movenet.model import MoveNetMultiPose
from movenet.losses import multipose_loss
from movenet.decode import decode_multipose


def main():
    torch.set_num_threads(2)
    model = MoveNetMultiPose(fpn_channels=32)
    x = torch.randn(2, 3, 128, 128)
    y = model(x); o = 32; n = 3
    batch = {
        "center_heatmap":torch.zeros(2,1,o,o), "keypoint_heatmap":torch.zeros(2,17,o,o),
        "keypoint_regression":torch.zeros(2,n,34), "center_offset":torch.zeros(2,n,2),
        "box_size":torch.zeros(2,n,2), "indices":torch.zeros(2,n,dtype=torch.long),
        "person_mask":torch.zeros(2,n), "keypoint_mask":torch.zeros(2,n,17),
        "keypoint_offset":torch.zeros(2,17,2,o,o), "keypoint_offset_mask":torch.zeros(2,17,o,o),
    }
    batch["center_heatmap"][:,:,8,8] = 1; batch["keypoint_heatmap"][:,:,8,8] = 1
    loss, _ = multipose_loss(y, batch); loss.backward()
    decoded = decode_multipose(y, max_people=2, center_threshold=0.)
    assert torch.isfinite(loss) and len(decoded) == 2 and len(decoded[0]) == 2
    print("smoke test passed", {k:tuple(v.shape) for k,v in y.items()}, "loss", float(loss))


if __name__ == "__main__": main()

