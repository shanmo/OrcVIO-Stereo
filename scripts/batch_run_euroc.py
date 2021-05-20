import os
import sys

#--------------------------------------------------------------------------------------------
# for EuRoC dataset start 
#--------------------------------------------------------------------------------------------

euroc_dataset_path = '/media/erl/disk2/euroc/'
euroc_csv_path = '/home/erl/dcist_vio_workspace/stereo_orcvio_ws/src/slam_groundtruth_comparison/data/euroc_mav/'

#--------------------------------------------------------------------------------------------
# for EuRoC dataset end 
#--------------------------------------------------------------------------------------------

orcvio_root_path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', ''))
euroc_cache_path = orcvio_root_path + "/cache/euroc/"
launch_file_name = "orcvio_euroc_eval.launch"
launch_file = orcvio_root_path + '/launch/' + launch_file_name
bag_dir = euroc_dataset_path

keywords_to_avoid = {
    "output_dir_traj", "output_dir_log", "rosbag"
}

# bag_name:bag_start in sec
bag_list = {
            'MH_01_easy':40,
            'MH_02_easy':35,
            'MH_03_medium':15,
            'MH_04_difficult':20,
            'MH_05_difficult':20,
            'V1_01_easy':0,
            'V1_02_medium':0,
            'V1_03_difficult':0,
            'V2_01_easy':0,
            'V2_02_medium':0
            }

run_cmd = 'roslaunch orcvio ' + launch_file_name

def run_once():
    os.system(run_cmd)
    return 

# [name:(type,value)]
def modify_launch(params):
    lines = open(launch_file, "r").readlines()
    fp = open(launch_file, "w")
    for line in lines:
        valid_flag = True 
        for name in params.keys():
            if name in line:
                for kw in keywords_to_avoid:
                    if kw in line:
                        valid_flag = False 
                if (valid_flag):
                    a, b = params[name]
                    if (name == 'path_gt'):
                        line = '    <param name="%s" type="%s" value="%s" />\n'%(name,a,b)
                    else: 
                        line = '    <arg name="%s" default="%s" />\n'%(name,b)
                    break
        fp.write(line)
    fp.close()

def loop_rosbag():
    params = {}
    for bag, start_ts in bag_list.items():
        fbag = os.path.join(bag_dir, bag + '.bag')
        fcsv = euroc_csv_path + "%s.csv"%bag
        fresult = os.path.join(euroc_cache_path, bag) + "/" 
        if os.path.exists(fbag):
            params['path_gt'] = ('string', fcsv)
            params['path_bag'] = ('string', fbag)
            params['bag_start'] = ('string', str(start_ts))
            params['path_traj'] = ('string', fresult)
            modify_launch(params)
            print("run {}".format(bag))
            res = run_once()
    return 

if __name__ == '__main__':

    # clean roslog
    os.system('rm -rf ~/.ros/log')

    loop_rosbag()