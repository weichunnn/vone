[Skip to main content](#main-content)

Back to top  

 Ctrl+K

[![DeepStream documentation - Home](../_static/nvidia-logo-horiz-rgb-blk-for-screen.svg)

DeepStream documentation

](../index.html)

*   [twitter](https://twitter.com/nvidiaomniverse "twitter")
*   [youtube](https://www.youtube.com/channel/UCSKUoczbGAcMld7HjpCR8OA "youtube")
*   [instagram](https://www.instagram.com/nvidiaomniverse "instagram")
*   [www](https://www.nvidia.com/en-us/omniverse/ "www")
*   [linkedin](https://www.linkedin.com/showcase/nvidia-omniverse "linkedin")
*   [twitch](https://www.twitch.tv/nvidiaomniverse "twitch")

# Quickstart Guide[#](#quickstart-guide "Link to this heading")

## Jetson \[Not applicable for NVAIE customers\][#](#jetson-not-applicable-for-nvaie-customers "Link to this heading")

This section explains how to use DeepStream SDK on a Jetson device.

### Boost the clocks[#](#boost-the-clocks "Link to this heading")

After you have installed DeepStream SDK, run these commands on the Jetson device to boost the clocks:

$ sudo nvpmodel \-m 0
$ sudo jetson\_clocks

You should run these commands before running DeepStream application.

### Run deepstream-app (the reference application)[#](#run-deepstream-app-the-reference-application "Link to this heading")

1.  Navigate to the configs/deepstream-app directory on the development kit.
    
    $ cd /opt/nvidia/deepstream/deepstream-7.1/samples/configs/deepstream-app
    
2.  Enter the following command to run the reference application:
    
    \# deepstream-app -c <path\_to\_config\_file>
    
    e.g. $deepstream\-app \-c source30\_1080p\_dec\_infer\-resnet\_tiled\_display\_int8.txt
    
    Where `<path_to_config_file>` is the pathname of one of the reference application’s configuration files, found in `configs/deepstream-app/`. See Package Contents in `configs/deepstream-app/` for a list of the available files.
    
    Config files that can be run with deepstream-app:
    
    1.  `source30_1080p_dec_infer-resnet_tiled_display_int8.txt`
        
    2.  `source30_1080p_dec_preprocess_infer-resnet_tiled_display_int8.txt`
        
    3.  `source4_1080p_dec_infer-resnet_tracker_sgie_tiled_display_int8.txt`
        
    4.  `source1_usb_dec_infer_resnet_int8.txt`
        
    5.  `source1_csi_dec_infer_resnet_int8.txt`
        
    6.  `source2_csi_usb_dec_infer_resnet_int8.txt`
        
    7.  `source6_csi_dec_infer_resnet_int8.txt`
        
    8.  `source2_1080p_dec_infer-resnet_demux_int8.txt`
        
    9.  `source4_1080p_dec_infer-resnet_tracker_sgie_tiled_display_int8.yml`
        
    10.  `source30_1080p_dec_infer-resnet_tiled_display_int8.yml`
        
    11.  `source4_1080p_dec_preprocess_infer-resnet_preprocess_sgie_tiled_display_int8.txt`
        
    12.  `source2_dewarper_test.txt`
        

Note

*   Refer to [Sample Configurations and Streams](DS_sample_configs_streams.html#ds-sample-configs-streams) for detailed explanation of each configuration file.
    

Note

> *   You can find sample configuration files under `/opt/nvidia/deepstream/deepstream-7.1/samples` directory. Enter this command to see application usage:
>     
>     $ deepstream\-app \--help
>     
> *   To save TensorRT Engine/Plan file, run the following command:
>     
>     $ sudo deepstream\-app \-c <path\_to\_config\_file\>
>     

1.  To show labels in 2D Tiled display view, expand the source of interest with mouse left-click on the source. To return to the tiled display, right-click anywhere in the window.
    
2.  Keyboard selection of source is also supported. On the console where application is running, press the `z` key followed by the desired row index (0 to 9), then the column index (0 to 9) to expand the source. To restore 2D Tiled display view, press `z` again.
    

### Expected output (deepstream-app)[#](#expected-output-deepstream-app "Link to this heading")

The image below shows the expected output of deepstream-app with `source30_1080p_dec_infer-resnet_tiled_display_int8.txt` config:

> ![DeepStream Reference Application Architecture](../_images/DS_reference_ds_app_expected_output_without_preprocess.png)

### Run precompiled sample applications[#](#run-precompiled-sample-applications "Link to this heading")

1.  Navigate to the chosen application directory inside `sources/apps/sample_apps`.
    
2.  Follow the directory’s README file to run the application.
    
    > Note
    > 
    > If the application encounters errors and cannot create Gst elements, remove the GStreamer cache, then try again. To remove the GStreamer cache, enter this command: `$ rm ${HOME}/.cache/gstreamer-1.0/registry.aarch64.bin`
    > 
    > When the application is run for a model which does not have an existing engine file, it may take up to a few minutes (depending on the platform and the model) for the file generation and the application launch. For later runs, these generated engine files can be reused for faster loading.
    

## dGPU for Ubuntu[#](#dgpu-for-ubuntu "Link to this heading")

This section explains how to use DeepStream SDK on a x86 machine.

### Run the deepstream-app (the reference application)[#](#run-the-deepstream-app-the-reference-application "Link to this heading")

*   Navigate to the configs/deepstream-app directory on the development kit.
    
    > $ cd /opt/nvidia/deepstream/deepstream-7.1/samples/configs/deepstream-app
    
*   Enter the following command to run the reference application:
    
    \# deepstream-app -c <path\_to\_config\_file>
    
    e.g. $deepstream\-app \-c source30\_1080p\_dec\_infer\-resnet\_tiled\_display\_int8.txt
    
    Where `<path_to_config_file>` is the pathname of one of the reference application’s configuration files, found in `configs/deepstream-app`. See Package Contents in `configs/deepstream-app/` for a list of the available files.
    
    Config files that can be run with deepstream-app:
    
    > 1.  `source30_1080p_dec_infer-resnet_tiled_display_int8.txt`
    >     
    > 2.  `source30_1080p_dec_preprocess_infer-resnet_tiled_display_int8.txt`
    >     
    > 3.  `source4_1080p_dec_infer-resnet_tracker_sgie_tiled_display_int8.txt`
    >     
    > 4.  `source4_1080p_dec_infer-resnet_tracker_sgie_tiled_display_int8_gpu1.txt`
    >     
    > 5.  `source1_usb_dec_infer_resnet_int8.txt`
    >     
    > 6.  `source2_1080p_dec_infer-resnet_demux_int8.txt`
    >     
    > 7.  `source4_1080p_dec_infer-resnet_tracker_sgie_tiled_display_int8.yml`
    >     
    > 8.  `source30_1080p_dec_infer-resnet_tiled_display_int8.yml`
    >     
    > 9.  `source4_1080p_dec_preprocess_infer-resnet_preprocess_sgie_tiled_display_int8.txt`
    >     
    > 10.  `source2_dewarper_test.txt`
    >     
    

Note

*   Refer to [Sample Configurations and Streams](DS_sample_configs_streams.html#ds-sample-configs-streams) for detailed explanation of each configuration file.
    

Note

*   To dump engine file, run the following command:
    
    $ sudo deepstream\-app \-c <path\_to\_config\_file\>
    
*   You can find sample configuration files under `/opt/nvidia/deepstream/deepstream-7.1/samples` directory. Enter this command to see application usage:
    
    $ deepstream\-app \--help
    

*   To show labels in 2D tiled display view, expand the source of interest with a mouse left-click on the source. To return to the tiled display, right-click anywhere in the window.
    
*   Keyboard selection of source is also supported. On the console where application is running, press the `z` key followed by the desired row index (0 to 9), then the column index (0 to 9) to expand the source. To restore the 2D Tiled display view, press `z` again.
    

### Expected output (deepstream-app)[#](#ds-quickstart-ds-app-x86 "Link to this heading")

The image below shows the expected output of deepstream-app with `source30_1080p_dec_infer-resnet_tiled_display_int8.txt` config:

> ![DeepStream Reference Application Architecture](../_images/DS_reference_ds_app_expected_output_without_preprocess.png)

### Run precompiled sample applications[#](#id2 "Link to this heading")

1.  Navigate to the chosen application directory inside `sources/apps/sample_apps`.
    
2.  Follow that directory’s README file to run the application.
    
    > Note
    > 
    > If the application encounters errors and cannot create Gst elements, remove the GStreamer cache, then try again. To remove the GStreamer cache, enter this command:
    > 
    > $ rm ${HOME}/.cache/gstreamer\-1.0/registry.x86\_64.bin
    > 
    > When the application is run for a model which does not have an existing engine file, it may take up to a few minutes (depending on the platform and the model) for the file generation and application launch. For later runs, these generated engine files can be reused for faster loading.
    

## How to visualize the output if the display is not attached to the system[#](#how-to-visualize-the-output-if-the-display-is-not-attached-to-the-system "Link to this heading")

### 1 . Running with an X server by creating virtual display[#](#running-with-an-x-server-by-creating-virtual-display "Link to this heading")

Refer [https://docs.nvidia.com/grid/latest/grid-vgpu-user-guide/index.html#configuring-xorg-server-on-linux-server](https://docs.nvidia.com/grid/latest/grid-vgpu-user-guide/index.html#configuring-xorg-server-on-linux-server) for details.

### 2 . Running without an X server (applicable for applications supporting RTSP streaming output)[#](#running-without-an-x-server-applicable-for-applications-supporting-rtsp-streaming-output "Link to this heading")

The default configuration files provided with the SDK have the EGL based `nveglglessink` as the default renderer (indicated by type=2 in the \[sink\] groups). The renderer requires a running X server and fails without one. In case of absence of an X server, DeepStream reference applications provide an alternate functionality of streaming the output over RTSP. This can be enabled by adding an RTSP out sink group in the configuration file. Refer to `[sink2]` group in `source30_1080p_dec_infer-resnet_tiled_display_int8.txt` file for an example. Don’t forget to disable the `nveglglessink` renderer by setting enable=0 for the corresponding sink group.

## DeepStream Triton Inference Server Usage Guidelines[#](#deepstream-triton-inference-server-usage-guidelines "Link to this heading")

To migrate the Triton version in a DeepStream 7.1 deployment (Triton 24.08) to a newer version (say Triton 24.09 or newer), follow the instructions at [DeepStream Triton Migration Guide](https://github.com/NVIDIA-AI-IOT/deepstream_triton_migration).

Note

*   Before running `prepare_classification_test_video.sh`, refer note in section [Docker Containers](DS_docker_containers.html#triton-dockers-ffmpeg-command).
    

### dGPU[#](#dgpu "Link to this heading")

1.  Pull the DeepStream Triton Inference Server docker
    
    docker pull nvcr.io/nvidia/deepstream:7.1\-triton\-multiarch
    
2.  Start the docker
    
    docker run \--gpus "device=0" \-it \--rm \-v /tmp/.X11\-unix:/tmp/.X11\-unix \-e DISPLAY\=$DISPLAY \-e CUDA\_CACHE\_DISABLE\=0 nvcr.io/nvidia/deepstream:7.1\-triton\-multiarch
    

Note

*   The triton docker for x86 & jetson is based on Tritonserver 24.08 docker and has Ubuntu 22.04.
    
*   When the triton docker is launched for the first time, it might take a few minutes to start since it has to generate compute cache.
    

### dGPU on ARM (IGX/dGPU, GH100, GH200, SBSA)[#](#dgpu-on-arm-igx-dgpu-gh100-gh200-sbsa "Link to this heading")

1.  Pull the DeepStream Triton Inference Server docker
    
    docker pull nvcr.io/nvidia/deepstream:7.1\-triton\-arm\-sbsa
    
2.  Start the docker
    
    sudo docker run \-it \--rm \--runtime\=nvidia \--network\=host \-e NVIDIA\_DRIVER\_CAPABILITIES\=compute,utility,video,graphics \--gpus all \--privileged \-e DISPLAY\=:0 \-v /tmp/.X11\-unix:/tmp/.X11\-unix \-v /etc/X11:/etc/X11 nvcr.io/nvidia/deepstream:7.1\-triton\-arm\-sbsa
    

Note

*   The triton docker for dGPU on ARM is based on Tritonserver 24.08 docker and has Ubuntu 22.04.
    
*   When the triton docker is launched for the first time, it might take a few minutes to start since it has to generate compute cache.
    
*   dGPU on ARM/SBSA docker currently supports only nv3dsink for video display via workaround mentioned in section [Known Limitation with Video Subsystem and Workaround](DS_Installation.html#ds-installation-sbsa-known-limitation).
    

### Jetson[#](#jetson "Link to this heading")

DeepStream Triton container image (nvcr.io/nvidia/deepstream:7.1-triton-multiarch) has Triton Inference Server and supported backend libraries pre-installed.

In order to run the Triton Inference Server directly on device, i.e., without docker, Triton Server setup will be required.

Go to samples directory and run the following commands to set up the Triton Server and backends.

$ cd /opt/nvidia/deepstream/deepstream/samples/
$ sudo ./triton\_backend\_setup.sh

Note

By default script will download the Triton Server version 2.49. For setting up any other version change the package path accordingly.

Triton backends are installed into `/opt/nvidia/deepstream/deepstream/lib/triton_backends` by default by the script. User can update `infer_config` settings for specific folders as follows:

model\_repo {
 backend\_dir: /opt/nvidia/tritonserver/backends/
}

## Using DLA for inference[#](#using-dla-for-inference "Link to this heading")

DLA is Deep Learning Accelerator present on the Jetson AGX Orin and Jetson Orin NX. These platforms have two DLA engines. DeepStream can be configured to run inference on either of the DLA engines through the Gst-nvinfer plugin. One instance of Gst-nvinfer plugin and thus a single instance of a model can be configured to be executed on a single DLA engine or the GPU. However, multiple Gst-nvinfer plugin instances can be configured to use the same DLA. To configure Gst-nvinfer to use the DLA engine for inference, modify the corresponding property in nvinfer component configuration file (example: samples/configs/deepstream-app/config\_infer\_primary.txt): Set enable-dla=1 in \[property\] group. Set use-dla-core=0 or use-dla-core=1 depending on the DLA engine to use.

DeepStream does support inferencing using GPU and DLAs in parallel. You can run this in separate processes or single process. You will need three separate sets of configs configured to run on GPU, DLA0 and DLA1:

### Separate processes[#](#separate-processes "Link to this heading")

When GPU and DLA are run in separate processes, set the environment variable `CUDA_DEVICE_MAX_CONNECTIONS` as `1` from the terminal where DLA config is running.

### Single process[#](#single-process "Link to this heading")

DeepStream reference application supports multiple configs in the same process. To run DLA and GPU in same process, set environment variable `CUDA_DEVICE_MAX_CONNECTIONS` as `32`:

$ deepstream\-app \-c <gpuconfig\> \-c <dla0config\> \-c <dla1config\>

On this page