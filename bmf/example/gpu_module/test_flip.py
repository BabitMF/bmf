import cvcuda, torch

cvimg_batch = cvcuda.ImageBatchVarShape(3)
cvimg_batch_out = cvcuda.ImageBatchVarShape(3)

# self.i420_in = hmp.Frame(in_frame.width, in_frame.height, self.i420info, device='cuda')
# hmp.img.yuv_to_yuv(self.i420_in.data(), in_frame.frame().data(), self.i420info, in_frame.frame().pix_info())
torch_in = [
    torch.empty((1280, 720, 1), dtype=torch.uint8, device='cuda'),
    torch.empty((640, 360, 1), dtype=torch.uint8, device='cuda'),
    torch.empty((640, 360, 1), dtype=torch.uint8, device='cuda')
]
torch_out = [
    torch.empty((1280, 720, 1), dtype=torch.uint8, device='cuda'),
    torch.empty((640, 360, 1), dtype=torch.uint8, device='cuda'),
    torch.empty((640, 360, 1), dtype=torch.uint8, device='cuda')
]
# in_list = [x.torch() for x in self.i420_in.data()]
# out_list = [x.torch() for x in self.i420_out.data()]
cvimg_batch.pushback([cvcuda.as_image(x) for x in torch_in])
cvimg_batch_out.pushback([cvcuda.as_image(x) for x in torch_out])

t3 = torch.ones((3, ), dtype=torch.int8, device='cuda')
cvt3 = cvcuda.as_tensor(t3)

cvcuda.flip_into(cvimg_batch_out, cvimg_batch, flipCode=cvt3)
