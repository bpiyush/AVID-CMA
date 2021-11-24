# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import glob
import numpy as np
from torch.functional import split


DATA_PATH = '/local-ssd/fmthoker/kinetics/VideoData'
ANNO_PATH = '/local-ssd/fmthoker/kinetics/labels'


from datasets.video_db import VideoDataset
class Kinetics(VideoDataset):
    def __init__(self, subset,
                 return_video=True,
                 video_clip_duration=1.,
                 video_fps=25.,
                 video_transform=None,
                 return_audio=False,
                 audio_clip_duration=1.,
                 audio_fps=None,
                 audio_fps_out=64,
                 audio_transform=None,
                 return_labels=False,
                 return_index=False,
                 max_offsync_augm=0,
                 mode='clip',
                 clips_per_video=1,
                 ):

        classes = sorted(os.listdir(f"{DATA_PATH}"))
 
        subset_prefix = "train" if subset == "train" else "val"
        subset_file = f"{ANNO_PATH}/{subset_prefix}_videofolder.txt"

        filenames = []
        labels = []
        for ln in open(subset_file):
            file_name, label = ln.strip().split()[0] + '.avi', int(ln.strip().split()[2])
            if os.path.isfile(os.path.join(DATA_PATH, file_name)):
                    filenames.append(file_name)
                    labels.append(label)
        
        print(":::::: Dataset details ::::::")
        print(f"Subset: {subset}")
        print(f"Number of samples: {len(filenames)}")

        super(Kinetics, self).__init__(
            return_video=return_video,
            video_root=DATA_PATH,
            video_fns=filenames,
            video_clip_duration=video_clip_duration,
            video_fps=video_fps,
            video_transform=video_transform,
            return_audio=return_audio,
            audio_root=DATA_PATH,
            audio_fns=filenames,
            audio_clip_duration=audio_clip_duration,
            audio_fps=audio_fps,
            audio_fps_out=audio_fps_out,
            audio_transform=audio_transform,
            return_labels=return_labels,
            labels=labels,
            return_index=return_index,
            mode=mode,
            clips_per_video=clips_per_video,
            max_offsync_augm=max_offsync_augm,
        )

        self.name = 'Kinetics dataset'
        self.root = DATA_PATH
        self.subset = subset

        self.classes = classes
        self.num_videos = len(filenames)
        self.num_classes = len(classes)

        self.sample_id = np.array([fn.split('/')[-1].split('.')[0].encode('utf-8') for fn in filenames])


if __name__ == '__main__':
    dataset = Kinetics('train', video_fps=16., video_clip_duration=1.)
    instance = dataset[0]

    assert "frames" in instance
    assert len(instance["frames"]) == 16

    # test with a real config
    from datasets import preprocessing

    db_cfg = {
        "name": "kinetics",
        "full_res": True,
        "batch_size": 256,
        "video_clip_duration": 1.0,
        "video_fps": 16,
        "crop_size": 112,
        "audio_clip_duration": 2,
        "audio_fps": 24000,
        "spectrogram_fps": 100,
        "n_fft": 512,
        "transforms": "msc+color",
    }
    split_cfg = {
        "split": "train",
        "use_augmentation": True,
        "drop_last": True,
        "clips_per_video": 10
    }
    num_frames = int(db_cfg['video_clip_duration'] * db_cfg['video_fps'])

    # video transforms
    video_transform = preprocessing.VideoPrep_MSC_CJ(
        crop=(db_cfg['crop_size'], db_cfg['crop_size']),
        augment=split_cfg['use_augmentation'],
        num_frames=num_frames,
        pad_missing=True,
    )

    # audio transforms
    audio_transforms = [
        preprocessing.AudioPrep(
            trim_pad=True,
            duration=db_cfg['audio_clip_duration'],
            augment=split_cfg['use_augmentation'],
            missing_as_zero=True),
        preprocessing.LogSpectrogram(
            db_cfg['audio_fps'],
            n_fft=db_cfg['n_fft'],
            hop_size=1. / db_cfg['spectrogram_fps'],
            normalize=True)
    ]
    audio_fps_out = db_cfg['spectrogram_fps']

    # define the dataset
    clips_per_video = split_cfg.get('clips_per_video', 1)
    db = Kinetics(
        subset=split_cfg['split'],
        return_video=True,
        video_clip_duration=db_cfg['video_clip_duration'],
        video_fps=db_cfg['video_fps'],
        video_transform=video_transform,
        return_audio=True,
        audio_clip_duration=db_cfg['audio_clip_duration'],
        audio_fps=db_cfg['audio_fps'],
        audio_fps_out=audio_fps_out,
        audio_transform=audio_transforms,
        max_offsync_augm=0.5 if split_cfg['use_augmentation'] else 0,
        return_labels=False,
        return_index=True,
        mode='clip',
        clips_per_video=clips_per_video,
    )
    instance = db[0]

    assert set(instance.keys()) == {'frames', 'audio', 'index'}
    assert instance["frames"].shape == (
        3, num_frames, db_cfg['crop_size'], db_cfg['crop_size'],
    )
    assert instance["audio"].shape == (
        1,
        db_cfg["spectrogram_fps"] * db_cfg["audio_clip_duration"],
        db_cfg['n_fft'] // 2 + 1,
    )
    assert instance["index"] == 0
