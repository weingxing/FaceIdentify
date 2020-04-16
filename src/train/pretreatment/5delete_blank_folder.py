import os

path = './star_image_processed'
dirs = os.listdir(path)
# print(dirs)

# 删除空文件夹
for dir in dirs[1:]:
    p = os.path.join(path, dir)
    n = len(os.listdir(p))
    if n == 0:
        print('%s 为空文件夹，删除...' % dir)
        os.removedirs(p)

print('清理完成')