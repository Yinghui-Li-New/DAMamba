import torch
import numpy as np
import matplotlib.pyplot as plt
import time

def ortho_render(pcl, rotmat_az, rotmat_el, resolution=50, npoints=2048, box_size=1.):
    pcl = torch.Tensor(pcl).cuda()
    rotmat_az = torch.Tensor(rotmat_az).cuda()
    rotmat_el = torch.Tensor(rotmat_el).cuda()
    pcl = torch.matmul(pcl, rotmat_az)
    pcl = torch.matmul(pcl, rotmat_el)
    depth = -box_size - pcl[:,2]
    grid_idx = (pcl[:,0:2] + box_size)/(2*box_size/resolution)
    grid_idx = grid_idx.long()
    grid_idx = torch.cat((grid_idx,torch.arange(npoints).view(npoints,-1).cuda()),1)
    grid_idx = grid_idx[:,0]*resolution*npoints + grid_idx[:,1]*npoints + grid_idx[:,2]
    plane_distance = torch.ones((resolution*resolution*npoints)).cuda() * -box_size*2
    plane_distance[grid_idx] = depth
    plane_distance = plane_distance.view(resolution,resolution,npoints)
    plane_depth,_ = torch.max(plane_distance,2)
    plane_mask = (plane_depth <= (-box_size * 2 + 1e-6))
    plane_mask = plane_mask.float() * box_size * 2
    plane_depth = plane_depth + plane_mask
    plane_depth -= box_size*2/50
    plane_depth = plane_depth.view(resolution,resolution,1)
    point_visible = (plane_distance >= plane_depth)
    point_visible, _ = torch.max(torch.max(point_visible.int(),0)[0],0)
    return point_visible.cpu().numpy()
def ortho_render_batch(pcl_batch, rotmat_az_batch, rotmat_el_batch, resolution=100, npoints=2048, box_size=1.):
    batch_size, npoints, dimension = pcl_batch.shape
    rotmat_az_batch = torch.Tensor(rotmat_az_batch).cuda()
    rotmat_el_batch = torch.Tensor(rotmat_el_batch).cuda()
    pcl_batch = torch.matmul(pcl_batch, rotmat_az_batch)
    pcl_batch = torch.matmul(pcl_batch, rotmat_el_batch)
    depth = -box_size - pcl_batch[:,:,2]
    grid_idx = (pcl_batch[:,:,0:2] + box_size)/(2*box_size/resolution)
    grid_idx = grid_idx.long()
    grid_idx = torch.cat((grid_idx,torch.arange(npoints).view(1,npoints,-1).cuda().repeat(batch_size, 1,1)),2)
    grid_idx = torch.cat(((torch.arange(batch_size).view(batch_size,1,-1).cuda().repeat(1, npoints, 1)),grid_idx), 2)
    grid_idx = grid_idx[:,:,0]*resolution*resolution*npoints + grid_idx[:,:,1]*resolution*npoints + grid_idx[:,:,2]*npoints + grid_idx[:,:,3]
    grid_idx = grid_idx.view(batch_size*npoints)
    device = torch.device('cuda:0')
    plane_distance = torch.ones((batch_size*resolution*resolution*npoints), device=device) * -box_size*2
    depth = depth.view(batch_size*npoints)
    plane_distance[grid_idx] = depth
    plane_distance = plane_distance.view(batch_size,resolution,resolution,npoints)
    plane_depth,_ = torch.max(plane_distance,3)
    plane_mask = (plane_depth <= (-box_size * 2 + 1e-6))
    plane_mask = plane_mask.float() * box_size * 2
    plane_depth = plane_depth + plane_mask
    plane_depth -= box_size*2/50 * 1
    plane_depth = plane_depth.view(batch_size,resolution,resolution,1)
    point_visible = (plane_distance >= plane_depth)
    point_visible,_ = torch.max(point_visible.int(),1)
    point_visible,_ = torch.max(point_visible.int(),1)
    return point_visible.cpu().numpy()
def partial_render_batch(pcl_batch, partial_batch, resolution=100, box_size=1.):    #gt, partial
    batch_size, npoints, dimension = pcl_batch.shape 
    rotmat_az_batch, rotmat_el_batch, az_batch, el_batch = generate_rotmat(batch_size)
    az_batch,el_batch = az_batch.reshape(batch_size,1), el_batch.reshape(batch_size,1)
    azel_batch = np.concatenate([az_batch,el_batch],1)
    point_visible_batch = ortho_render_batch(pcl_batch, rotmat_az_batch, rotmat_el_batch).reshape(batch_size,npoints,1)
    for i in range(batch_size):
        point_visible = point_visible_batch[i,:,:]
        pcl = pcl_batch[i,:,:]
        point_visible_idx, _ = np.where(point_visible > 0.5)
        point_visible_idx = np.random.choice(point_visible_idx, 2048)
        new_pcl = pcl[point_visible_idx]
        partial_batch[i,:,:] = new_pcl

    return partial_batch , rotmat_az_batch, rotmat_el_batch, azel_batch


def generate_rotmat(batch_size):
    az_batch = np.random.rand(batch_size)
    az_batch = az_batch * 2 * np.pi
    el_batch = np.random.rand(batch_size)
    el_batch = (el_batch - 0.5) * np.pi
    rotmat_az_batch = np.array([
        [np.cos(az_batch),     -np.sin(az_batch),    np.zeros(batch_size)],
        [np.sin(az_batch),     np.cos(az_batch),     np.zeros(batch_size)],
        [np.zeros(batch_size), np.zeros(batch_size), np.ones(batch_size)]]
        )
    rotmat_az_batch = np.transpose(rotmat_az_batch, (2,0,1)) 
    rotmat_el_batch = np.array([
        [np.ones(batch_size),  np.zeros(batch_size), np.zeros(batch_size)],
        [np.zeros(batch_size), np.cos(el_batch),     -np.sin(el_batch)],
        [np.zeros(batch_size), np.sin(el_batch),     np.cos(el_batch)]]
        )
    rotmat_el_batch = np.transpose(rotmat_el_batch, (2,0,1)) 
    return rotmat_az_batch, rotmat_el_batch, az_batch, el_batch


if __name__ == "__main__":
    def visualize_pc(points):
        xs = points[:,0]
        ys = points[:,1]
        zs = points[:,2]
        fig = plt.figure()
        ax=fig.add_subplot(projection='3d')

        ax.scatter(xs,ys,zs,marker='o')
        ax.set_xlim(-0.5,0.5)
        ax.set_ylim(-0.5,0.5)
        ax.set_zlim(-0.5,0.5)
        plt.show()
    pcl1 = np.load("datasets/ModelNet40_Completion/table/train/table_0294_3_complete.npy")
    pcl2 = np.load("datasets/ModelNet40_Completion/table/train/table_0290_3_complete.npy")
    pcl_batch = np.concatenate((pcl1.reshape(1,2048,3),pcl2.reshape(1,2048,3)),axis=0)
    pcl_batch = np.repeat(pcl_batch, 5, axis=0)
    pcl_batch = torch.Tensor(pcl_batch).cuda()
    batch_size, npoints, dimension = pcl_batch.shape
    partial_batch = torch.Tensor(np.zeros((batch_size,npoints,dimension))).cuda()

    new_pcl_batch, _, _, _ = partial_render_batch(pcl_batch, partial_batch)

    new_pcl_batch = new_pcl_batch.cpu().numpy()
    for i in range(10):
        new_pcl = new_pcl_batch[i]
        visualize_pc(new_pcl)


