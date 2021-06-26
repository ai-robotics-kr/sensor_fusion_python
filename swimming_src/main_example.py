import os
import sys

sys.path.insert()

from dataset_mgmt.kitti_manager import KittiDatasetMgmt
from filter.h_inf import HInf

kitti_root_dir = "/home/kimsooyoung/Documents/AI_KR"
kitti_date = "2011_09_30"
kitti_drive = "0033"

if __name__ == "__main__":

    dataset_mgmt = KittiDatasetMgmt(kitti_root_dir, kitti_date, kitti_drive)

    # plot your dataset
    dataset_mgmt.plotGPStrajactory()
    dataset_mgmt.plotXYZtrajactory()
    dataset_mgmt.plotGTvalue()

    # Noise Addition!!
    dataset_mgmt.addGaussianNoiseToGPS()

    # Plot Noise-Added Data
    dataset_mgmt.plotNoisytrajactory()

    try:
        pass
        # HInf()
    except Exception as e:
        print(e)
    finally:
        print(f"Done...")
