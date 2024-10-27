import os
import glob
import argparse
import json

import cv2
import openslide
import numpy as np
import pandas as pd
from datetime import datetime

from tiler.wsi_parser import WSIParser
from tiler.utilities import TissueDetect, visualise_wsi_tiling, StainNormalizer


def parse_wsi(args, wsi_path):
    
    try:
        wsi = openslide.OpenSlide(wsi_path)
    except openslide.lowlevel.OpenSlideError as e:
        return False
    
    try:
        args.base_mag = wsi.properties[
            openslide.PROPERTY_NAME_OBJECTIVE_POWER]
    except:
        pass
    
    try:
        args.mpp = wsi.properties[
            openslide.PROPERTY_NAME_MPP_X]
    except:
        pass

    print(f'Base mag: {args.base_mag}, mpp: {args.mpp}')
    #ds = int(int(args.base_mag) / 10) 
    ds_factors = [int(d) for d in wsi.level_downsamples]
    level = ds_factors.index(32) if 32 in ds_factors else ds_factors.index(int(ds_factors[-1]))
    #level = ds_factors.index(ds)
    #print(f'Downsample: {ds} \nLevel: {level}')
    detector = TissueDetect(wsi)
    thumb = detector.tissue_thumbnail
    tis_mask = detector.detect_tissue()
    border = detector.border()

    cv2.imwrite(os.path.join(
        args.vis_path, args.name+'_thumb.png'), thumb)
    cv2.imwrite(os.path.join(
        args.vis_path, args.name+'_mask.png'), tis_mask)

    if args.normalize:
        normalizer = StainNormalizer(args.sn_target, args.sn_method)
    else:
        normalizer = None

    #parser = WSIParser(wsi, args.tile_dims, border, args.mag_level)
    parser = WSIParser(wsi, args.tile_dims, border, args.mag_level, normalizer)

    num = parser.tiler(args.stride)
    print('Tiles: {}'.format(num))

    parser.filter_tissue(
        tis_mask,
        label=1,
        threshold=args.filter_threshold)
    print(f'Filtered tiles: {parser.number}')

    visualise_wsi_tiling(
            wsi,
            parser,
            os.path.join(args.vis_path,
                         args.name+'_tiling.png'),
            viewing_res=level
            )
     
    if args.sample:
        parser.sample_tiles(args.sample)
        #print('tilessssss',len(parser._tiles))
        print(f'Sampled tiles: {parser.number}')

    if args.parser != 'tiler':
        func = parser.extract_features(
            args.parser,
            args.model_path,
            downsample=args.downsample,
            normalize=args.normalize
        )
        #print("Feature extracted!")
    else:
        func = parser.extract_tiles(args.normalize)

    if args.parser != 'tiler' and args.database == 'lmdb':
        parser.to_lmdb(
            func,
            os.path.join(args.tile_path, args.name), 
            map_size = args.map_size)
        print("parsed to lmdb")
    elif args.parser != 'tiler' and args.database == 'rocksdb':
            parser.to_rocksdb(func, os.path.join(args.tile_path, args.name))
            print("parsed to rocksdb")
    elif args.parser != 'tiler' and args.database == 'disk':
        parser.feat_to_disk(func, os.path.join(args.tile_path, args.name))
        print("parsed to feat_to_disk")
    else:
        parser.save(func, os.path.join(args.tile_path, args.name))
    
    return True
    
    
if __name__=='__main__':

    ap=argparse.ArgumentParser()

    ap.add_argument('-wp','--wsi_path',
            required=True, help='whole slide image directory')

    ap.add_argument('-sp','--save_path',
            required=True, help='directoy to write tiles and features')

    ap.add_argument('-p', '--parser',
            required=True, help='wsi parsing approach')

    ap.add_argument('-mp', '--model_path', default=None,
            help='path to trained model if parser is not tiler')

    ap.add_argument('-td','--tile_dims', default=512, type=int,
            help='dimensions of tiles')

    ap.add_argument('-s','--stride', default=512, type=int,
            help='distance to step across WSI')

    ap.add_argument('-ml','--mag_level', default=0, type=int,
            help='magnification level of tiling')

    ap.add_argument('-ds','--downsample', required=False, type=int,
            help='downsample tiles to resolution inbetween levels')

    ap.add_argument('-n','--normalize', default=False, type=bool,
            help='downsample tiles to resolution inbetween levels')

    ap.add_argument('-sm', '--sn_method', default='macenko',
            help='Stain normalization method (options: macenko, vahadane, reinhard)')

    ap.add_argument('-st', '--sn_target', default=None,
            help='Path to the target image for stain normalization')

    ap.add_argument('-sa','--sample', required=False, type=int,
            help='sample number')

    ap.add_argument('-db','--database', default=None,
            help='store tiles/features in database. Options are lmdb, rocksdb, or disk')

    ap.add_argument('-mz', '--map_size', default=int(4e9), type=int,
            help='map_size of lmdb database if in use')

    ap.add_argument('-ft','--filter_threshold', default=0.5, type=float,
            help='threshold to filter tiles based on background')

    ap.add_argument('-dp','--dataset_csv', default=None,
            help='Path to CSV of slides and corresponding labels')

    args=ap.parse_args()

    dir_ = 'tiles' if args.parser == 'tiler' else 'features'
    args.tile_path = os.path.join(args.save_path, dir_)
    args.vis_path = os.path.join(args.save_path, 'vis')
    os.makedirs(args.tile_path, exist_ok=True)
    os.makedirs(args.vis_path, exist_ok=True)
    
    # Print arguments
    print("\nParsed arguments:\n")

    for k, v in vars(args).items():
        print(f"{k}: {v}")
    
    print("\n")
   
    # Save parsed arguments as json for future reference
    json_file_path = os.path.join(args.save_path, 'arguments.json')
    with open(json_file_path, 'w') as json_file:
        json.dump(vars(args), json_file, indent=4)
    
    # If dataset csv is supplied
    if args.dataset_csv is not None:
        saved = pd.read_csv(args.dataset_csv)
        if 'ID' in saved.columns:
            if saved['ID'].notna().any():
                saved = list(saved['ID'])
            else:
                print(f"Column 'ID' exists in {args.dataset_csv} but is empty.")
        else:
            print(f"Column 'ID' does not exist in {args.dataset_csv}.")

    errors = []
    success = dict(name = [], time = [])

    wsi_paths=glob.glob(os.path.join(args.wsi_path,'*'))

    for f in wsi_paths:
        path, ext = os.path.splitext(f)
        name = os.path.basename(path)
  
        if os.path.exists(os.path.join(args.tile_path,name)) and os.listdir(os.path.join(args.tile_path,name)):
            #print('continue')
            print(f'{name} already exists and not empty. Skipping...')
            continue
        
        print('wsi_name', name)
        
        args.name = name
        print(f'Parsing {name}...')
        status = parse_wsi(args, f)
        print(f'Parsed {status}')
        if not status:
            errors.append(name)
        else:
            success["name"].append(name)
            success["time"].append(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    err_df = pd.DataFrame({'name': errors})
    err_df.to_csv(os.path.join(args.save_path,'errors.csv'))

    succ_df = pd.DataFrame(success)
    succ_df.to_csv(os.path.join(args.save_path,'success_log.csv'))
    
    print('Finished parsing')



