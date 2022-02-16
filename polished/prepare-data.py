#!/usr/bin/env python
# coding: utf-8

# In[7]:


import keras
import cv2
import os
import glob
import random
import numpy as np
import pandas as pd
import cmder


# slipt one video to multiple segments.
def splitVideo(path: str, video: str, split: int) -> str:
    video_name = video.split(".")[0]
    newdir = path + video_name + "-splitby-" + str(split) + "-seconds"
    if os.path.exists(newdir):
        cmder.runCmd(f"rm -fr {newdir}")
    cmder.runCmd(f"mkdir -p {newdir}")
    command = (
        "ffmpeg -i "
        + path
        + video
        + " -reset_timestamps 1 -map 0 -segment_time "
        + str(split)
        + " -f segment "
        + newdir
        + "/"
        + video[:-4]
        + "%03d.mp4"
    )
    cmder.runCmd(command)
    return newdir + "/"


# extract images from videos into new directories.
def extractImages(path: str, video: str, split: int, dest: str) -> str:
    video_name = video.split(".")[0]
    newdir = path + video_name + "-extractby-" + str(split) + "-seconds/" + dest
    if os.path.exists(newdir):
        cmder.runCmd(f"rm -fr {newdir}")
    cmder.runCmd(f"mkdir -p {newdir}")
    command = (
        "ffmpeg -i "
        + path
        + video
        + " -vf fps=30 "
        + newdir
        + "/"
        + video[:-4]
        + "%d.png"
    )
    cmder.runCmd(command)
    return "/".join(newdir.split("/")[:-1])


# make both saliency and motion map videos as numpy arrays.
def makeVideoNumpy(fps: int, split: int, video: str, path: str):
    totalFrames = 1800
    one_batch_images = fps * split  # total images to read based on split
    extract_dirs = os.listdir(path)
    for extract_dir in extract_dirs:
        extract_dir = os.path.abspath(extract_dir) + "/"
        newdir = extract_dir + "Images"
        count = 1
        for i in range(1, totalFrames + 1, one_batch_images):
            npArray = newdir + "/" + video[:-4] + str(count) + ".npy"
            count += 1
            tempArray = []
            for j in range(one_batch_images):
                imageName = newdir + "/" + video[:-4] + str(i + j) + ".png"
                image = cv2.imread(imageName, cv2.IMREAD_GRAYSCALE)
                tempArray.append(image)
            print(np.array(tempArray).shape)
            np.save(npArray, np.array(tempArray))


if __name__ == "__main__":
    fps = 30
    split = 5
    path = "../datasets/content/videos/"
    video = "070.mp4"
    saliency = "saliency"
    motion = "motion"
    root = extractImages(path, video, split, saliency)
    cmder.runCmd(f"cp -r {root}/{saliency} {root}/{motion}")


# makeVideoNumpy(saliVideos, split, "saliency")
# makeVideoNumpy(motionVideos, split, "motion")
#
#
# # In[133]:
#
#
# make tiles information as numpy arrays
def makeTilesNumpyBackup(videosTileData, split):
    files = glob.glob("../datasets/sensory/tile/*01*.csv")
    files = sorted(files)[5:]
    fps = 30
    totalFrames = 1800
    imagesToRead = fps * split  # total images to read based on split
    for video, f in zip(videosTileData, files):
        count = 1
        for i in range(1, totalFrames + 1, imagesToRead):
            npArray = f[:-8] + str(count) + "_split_by_" + str(split) + "_seconds.npy"
            tempArray = []
            for j in range(imagesToRead):
                tempArray.append(video[i + j - 1])
            # print(np.array(tempArray).shape)
            np.save(npArray, np.array(tempArray))
            count += 1


# make tiles information as numpy arrays
def makeTilesNumpy(video, f, split):
    fps = 30
    totalFrames = 1800
    imagesToRead = fps * split  # total images to read based on split
    count = 1
    for i in range(1, totalFrames + 1, imagesToRead):
        npArray = f[:-8] + str(count) + ".npy"
        tempArray = []
        for j in range(imagesToRead):
            tempArray.append(video[i + j - 1])
        # print(np.array(tempArray).shape)
        np.save(npArray, np.array(tempArray))
        count += 1


# encode tile information into images in banary fashion for convenience
def encodeTileInfoToImages():
    files = glob.glob("../datasets/sensory/tile/*.csv")
    for f in files:
        width = 240
        breadth = 120
        tileSize = 12
        frames = []
        tilesInColumn = width / tileSize
        lines = open(f, "r")
        lines = lines.readlines()[1:]
        tilesInFrames = []
        for line in lines:
            line = line.strip().split(",")[1:]
            line = [int(i) for i in line]
            tilesInFrames.append(line)
        for i, tiles in enumerate(tilesInFrames):
            frame = np.zeros(width * breadth, dtype=bool)
            for tileNo in tiles:
                tileRowNumber = int((tileNo - 1) / tilesInColumn)
                tileColumnNumber = (tileNo - 1) % tilesInColumn
                firstPixel = (
                    tileRowNumber * width * tileSize + tileColumnNumber * tileSize
                )
                for rowPixel in range(0, tileSize):
                    for columnPixel in range(0, tileSize):
                        frame[int(firstPixel + rowPixel * width + columnPixel)] = 1
            frame = frame.reshape((120, 240))
            frames.append(frame)
        makeTilesNumpy(frames, f, split)
        frames = None


#
#
# encodeTileInfoToImages()
#
#
# # In[5]:
#
#
# files = glob.glob("../datasets/sensory/tile/*.csv")
# print(len(files))
#
#
# # In[28]:
#
#
# p = "../datasets/content/saliencyImages/landscape_saliency1.npy"
# data = np.load(p)
# print(data.shape)
#
#
# # In[50]:
#
#
# width = 512
# breadth = 512
# tiles = [1, 14, 19, 12]
# tileSize = 64
# frame = np.zeros(width * breadth)
# tilesInColumn = width / tileSize
#
# for tileNo in tiles:
#     tileRowNumber = int((tileNo - 1) / tilesInColumn)
#     tileColumnNumber = (tileNo - 1) % tilesInColumn
#     firstPixel = tileRowNumber * width * tileSize + tileColumnNumber * tileSize
#     for rowPixel in range(0, tileSize):
#         for columnPixel in range(0, tileSize):
#             frame[int(firstPixel + rowPixel * width + columnPixel)] = 255
#
# # frame = np.reshape(frame, (breadth, width))width*
# # print(frame)
#
#
# # In[ ]:
#
#
# from PIL import Image
# import numpy as np
# from matplotlib import pyplot as plt
#
# plt.imshow(frame, interpolation="nearest")
# plt.savefig("sample.png")
#
#
# # In[20]:
#
#
# def makeTilesProbNumpy(video, f, split):
#     fps = 30
#     totalFrames = 1800
#     imagesToRead = fps * split  # total images to read based on split
#     count = 1
#     for i in range(1, totalFrames + 1, imagesToRead):
#         npArray = "../datasets/sensory/tileProb/" + f[25:-8] + str(count) + ".npy"
#         tempArray = []
#         for j in range(imagesToRead):
#             tempArray.append(video[i + j - 1])
#         # print(np.array(tempArray).shape)
#         np.save(npArray, np.array(tempArray))
#         count += 1
#
#
# # In[21]:
#
#
# files = glob.glob("../datasets/sensory/tile/*.csv")
# newdir = "../datasets/sensory/tileProb"
# if os.path.exists(newdir):
#     os.system("rm -rf " + newdir)
# os.makedirs(newdir)
# for f in files:
#     lines = open(f, "r")
#     lines = lines.readlines()[1:]
#     frames = np.zeros((1800, 200))
#     count = 0
#     for line in lines:
#         line = line.strip().split(",")[1:]
#         line = [int(i) for i in line]
#         for i in line:
#             frames[count][i - 1] = 1
#         count += 1
#     makeTilesProbNumpy(frames, f, 1)
#
#
# # In[14]:
#
#
# f = "../datasets/sensory/tile/ride_user44_tile.csv"
# print("../datasets/sensory/tileProb/" + f[25:-8] + str(count) + ".npy")
#
#
# # In[28]:
#
#
# ## Separate the data into Train and Test folders
# files = glob.glob("../datasets/sensory/tile/*.csv")
# random.shuffle(files)
#
#
# # In[34]:
#
#
# train = files[:400]
# test = files[400:]
# for i in test:
#     for j in range(1, 61):
#         os.system("mv " + i[:-8] + str(j) + ".npy " + "../datasets/testData/sens/")
#
#
# # In[ ]:
#
#
# # In[ ]:
