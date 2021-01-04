def grid_sampler_unnormalize(coord, side, align_corners):
    if align_corners:
        return ((coord + 1) / 2) * (side - 1)
    else:
        return ((coord + 1) * side - 1) / 2
        
def grid_sampler_compute_source_index(coord, size, align_corners):
    coord = grid_sampler_unnormalize(coord, size, align_corners)
    return coord

def safe_get(image, n, c, x, y, H, W):
    value = torch.Tensor([0])
    if  x >= 0 and x < W and y >=0 and y < H:
        value = image[n, c, y, x]
    return value

    
def bilinear_interpolate_torch_2D(image, grid, align_corners=False):
    '''
         input shape = [N, C, H, W]
         grid_shape  = [N, H, W, 2]
    
         output shape = [N, C, H, W]
    '''
    N, C, H, W = image.shape
    grid_H = grid.shape[1]
    grid_W = grid.shape[2]
    
    output_tensor = torch.zeros_like(image)
    for n in range(N):
        for w in range(grid_W):
            for h in range(grid_H):
                #get corresponding grid x and y
                x = grid[n, h, w, 1]
                y = grid[n, h, w, 0]
                
                #Unnormalize with align_corners condition
                ix = grid_sampler_compute_source_index(x, W, align_corners)
                iy = grid_sampler_compute_source_index(y, H, align_corners)
                
                x0 = torch.floor(ix).type(torch.LongTensor)
                x1 = x0 + 1

                y0 = torch.floor(iy).type(torch.LongTensor)
                y1 = y0 + 1
    
                #Get W matrix before I matrix, as I matrix requires Channel information
                wa = (x1.type(torch.FloatTensor)-ix) * (y1.type(torch.FloatTensor)-iy) 
                wb = (x1.type(torch.FloatTensor)-ix) * (iy-y0.type(torch.FloatTensor)) 
                wc = (ix-x0.type(torch.FloatTensor)) * (y1.type(torch.FloatTensor)-iy) 
                wd = (ix-x0.type(torch.FloatTensor)) * (iy-y0.type(torch.FloatTensor)) 
                
                #Get values of the image by provided x0,y0,x1,y1 by channel
                for c in range(C):
                    #image, n, c, x, y, H, W
                    Ia = safe_get(image, n, c, y0, x0, H, W)
                    Ib = safe_get(image, n, c, y1, x0, H, W)
                    Ic = safe_get(image, n, c, y0, x1, H, W)
                    Id = safe_get(image, n, c, y1, x1, H, W)
                    out_ch_val = torch.t((torch.t(Ia)*wa)) + torch.t(torch.t(Ib)*wb) + \
                                          torch.t(torch.t(Ic)*wc) + torch.t(torch.t(Id)*wd)

                    output_tensor[n, c, h, w] = out_ch_val
    return output_tensor
