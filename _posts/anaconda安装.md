# 安装anaconda软件过程

1. 确定需要安装的目录，将路径转到相应目录下

2. 下载anaconda安装包

   ```shell
   wget https://repo.continuum.io/archive/Anaconda3-5.0.1-Linux-x86_64.sh
   ```

   ![](https://raw.githubusercontent.com/Thornhill-GYL/markdownpicture/master/%E5%AE%89%E8%A3%85%E5%8C%85%E4%B8%8B%E8%BD%BD%E5%AE%8C%E6%88%90.png)

3. 下载完成后直接安装

   ```shell
   bash Anaconda3-5.0.1-Linux-x86_64.sh
   ```

4. 过程中有不断让你确定的事项，一直enter就行，会出现installing，如下图

   ![](https://raw.githubusercontent.com/Thornhill-GYL/markdownpicture/master/%E5%AE%89%E8%A3%85%E8%BF%87%E7%A8%8B.png)

5. 最后要确认路径在用户目录下执行

   ```shell
   source .bashrc
   ```

6. 安装成功

   ![1570936502434](C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\1570936502434.png)

7. 保持更新可输入

   ```shell
   source ~/anaconda3/bin/activate root
   conda upgrade --all
   ```



