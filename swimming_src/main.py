from dataset_mgmt.kitti_manager import KittiDatasetMgmt
from filter.h_inf import HInf

kitti_root_dir = "/home/kimsooyoung/Documents/AI_KR"
kitti_date = "2011_09_30"
kitti_drive = "0033"

if __name__ == "__main__":
    try:
        KittiDatasetMgmt(kitti_root_dir, kitti_date, kitti_drive)
        HInf()
    except Exception as e:
        print(e)
    finally:
        print(f"Done...")
