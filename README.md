# VideoExtractWin


## What's this：このプログラムについて

This program aims to detect wildlife from camera trap images using [MegaDetector (Beery et al. 2019)](https://github.com/microsoft/CameraTraps) and to extract videos in which animals were detected. This document is a minimal description and will be updated as needed.  
このプログラムは、[MegaDetector (Beery et al. 2019)](https://github.com/microsoft/CameraTraps)を利用してカメラトラップ映像から野生動物を検出し、動物が検出された動画を抽出することを目的として作成されました。このドキュメントは現時点では最低限の記述しかされていないため、今後随時更新していく予定です。

 

---

## Get Started：はじめに

<br />

### Prerequisites：環境整備

* OS  
    The following code was tested on Windows 10 Pro.  
    During the test run, .jpg as the image file format.  
    以下のコードはWindows 10 Proで動作確認しています。  
    動作確認時、動画ファイル形式は.mp4を用いました。


* NVIDIA Driver
    NVIDAドライバーをインストールする

    Please refer to [NVIDIA Driver Version Check](https://www.nvidia.com/Download/index.aspx?lang=en-us).
    *** is a placeholder. Please enter the recommended nvidia driver version.  
    [NVIDIAドライババージョンチェック](https://www.nvidia.com/Download/index.aspx?lang=en-us)を参照し、***に推奨されるnvidiaドライババージョンを入力した上で実行してください。  

    Check installation.  
    インストール状況の確認。

    ```commandprompt
    nvidia-smi 
    # NVIDIA Driver installation check
    ```

        If nvidia-smi does not work, Try Rebooting.  
        nvidia-smiコマンドが動作しない場合は再起動してみてください。

* Conda

    Download installer and run the script.  
    インストーラーをダウンロードしてスクリプトを実行します。

    condaのパスを通す
    システム環境変数の編集->環境変数->PATH->新規->condaのpathをコピペ


<br />

### Instllation：インストール

1. Clone the Repository：リポジトリの複製

    Run ```git clone```,  
    ```git clone```を実行する


    or Download ZIP and Unzip in any directory of yours. The following codes are assumed that it was extracted to the user's home directory (`/home/${USER}/`).  
    もしくはZIPをダウンロードし、任意のディレクトリで解凍してください。なお、このページではユーザのホームディレクトリ（`/home/${USER}/`）に解凍した前提でスクリプトを記載しています。

2. Move Project Directory：プロジェクトディレクトリへ移動

    ```commandprompt
    cd {VideExtractWinのパス}
    ```

3. create conda environment：conda環境の構築

    ```commandprompt
    conda env create -f environment.yml
    conda activate pwlife
    ```
4. 以下のサイトを見てバージョンを合わせたものをインストールする
    CUDA Toolkit 11.3 Downloads
    https://developer.nvidia.com/cuda-downloads 

    cudnnのインストール(ログインが必要)
    https://developer.nvidia.com/rdp/cudnn-download

  
5. PytorchWildlifeのインストール
    Install through pip:
    ```commandprompt
    pip install PytorchWildlife
    ```
6.　バージョン管理
    ```commandprompt
    pip uninstall torch
    conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
    pip uninstall -y charset_normalizer
    pip install charset_normalizer==2.0.0
    ```
<br />



---

## Usage：使い方

<br />

0. ディレクトリの移動

    ```commandprompt
    cd {VideoExtractWinのパス}
    ```
    
1. conda環境のアクティベート

    ```commandprompt
    conda activate pwlife
    ```


2. gpuが使えるか確認

    ```commandprompt(conda)
    python gpu_check.py
    ```


3. Run MegaDetector  
  MegaDetectorの実行

    ```commandprompt(conda)
    python exec_mdet.py {カメラデータが入ったフォルダ}
    ```  

    {カメラデータが入ったフォルダ}_outにcsvファイルとjsonファイルが保存される



