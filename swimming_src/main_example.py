from dataset_mgmt import KittiDatasetMgmt
from filter import EKF

kitti_root_dir = "/Users/swimm_kim/Documents/Dataset"
# kitti_root_dir = "/home/kimsooyoung/Documents/AI_KR"
kitti_date = "2011_09_30"
kitti_drive = "0033"

PLOT_DATA = True

if __name__ == "__main__":

    dataset_mgmt = KittiDatasetMgmt(kitti_root_dir, kitti_date, kitti_drive)

    # plot your dataset
    if PLOT_DATA is True:
        dataset_mgmt.plotGPStrajactory()
        dataset_mgmt.plotXYZtrajactory()
        dataset_mgmt.plotGTvalue()

    # Noise Addition!!
    dataset_mgmt.addGaussianNoiseToGPS()

    # Plot Noise-Added Data
    if PLOT_DATA is True:
        dataset_mgmt.plotNoisytrajactory()

    try:
        # ekf = EKF()
        pass
    except Exception as e:
        print(e)
    finally:
        print(f"Done...")
