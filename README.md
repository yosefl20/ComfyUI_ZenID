# 🌟 ZenID: Face Swap 🌟
Inspired by [**InstantID**](https://github.com/instantX-research/InstantID) and [**InstantID Comfy**](https://github.com/cubiq/ComfyUI_InstantID)

This **ZenID Node** has been refactored for specialized tasks like _Face Swap_

**✨ Support further development by starring the project! ✨**

## 📑 **Table of Contents** 
1. [Updates](#updates) 
2. [ZenID Features](#zenid-features) 
    * [Face Swap](#zenid-face-swap) 
    * [Face Combine](#zenid-face-combine) 
3. [Installation](#installation)

## 📅 **Updates** <a name="updates"></a> 
* **2024/11/19**: Cập nhật README Tiếng Việt [READMEVN](https://github.com/vuongminh1907/ComfyUI_ZenID/blob/main/README_VN.md)
* **2024/11/18**: Released the new **_Face Swap_** functionality.
* **2024/11/16**: Added the **_Face Combine_** feature. 

## 🌟 **ZenID Features** <a name="zenid-features"></a>

### 🔗 **ZenID Face Swap** <a name="zenid-face-swap"></a>
- **Workflows**
    Sample [`ZenID_FaceSwap.json`](https://github.com/vuongminh1907/ComfyUI_ZenID/blob/main/workflow/ZenID_FaceSwap.json) are included in the `workflow` folder.
    ![ZenID Face Swap Example](https://github.com/vuongminh1907/ComfyUI_ZenID/blob/main/examples/zenid_faceswap.png)

- **Examples**
    - **Source Images**  
      <img src="https://github.com/vuongminh1907/ComfyUI_ZenID/blob/main/examples/blackwukong.png" width="300" /> 
      <img src="https://github.com/vuongminh1907/ComfyUI_ZenID/blob/main/examples/domixi.png" width="300" />

    - **Result Image**  
       <img src="https://github.com/vuongminh1907/ComfyUI_ZenID/blob/main/examples/result_faceswap.png" width="600" />


### 🔗 **ZenID Face Combine** <a name="zenid-face-swap"></a>
- **Workflows**

    Sample [`ZenID_combineFace.json`](https://github.com/vuongminh1907/ComfyUI_ZenID/blob/main/workflow/ZenID_combineFace.json) are included in the `workflow` folder.
    ![ZenID Face Combine Example](https://github.com/vuongminh1907/ComfyUI_ZenID/blob/main/examples/zenid_combineface.png)
- **Examples**
    - **Source Images**  
      <img src="https://github.com/vuongminh1907/ComfyUI_ZenID/blob/main/examples/haitu.jpg" width="300" /> 
      <img src="https://github.com/vuongminh1907/ComfyUI_ZenID/blob/main/examples/sontung.jpg" width="300" />

    - **Result Image**  
       <img src="https://github.com/vuongminh1907/ComfyUI_ZenID/blob/main/examples/result_facecombine.png" width="600" />

## ⚙️ Installation <a name="installation"></a>
1. Upgrade ComfyUI to the latest version.
```
git clone https://github.com/comfyanonymous/ComfyUI
```
2. Clone or download this repository into the `ComfyUI/custom_nodes/` directory.
```
cd ComfyUI/custom_nodes/
git clone https://github.com/vuongminh1907/ComfyUI_ZenID
```
3. Download the model
```
pip install -r ComfyUI_ZenID/requirements.txt
python ComfyUI_ZenID/downloadmodel.py
```
4. Run ComfyUI 

## Contact for Work 🌟

This is a demo Face Swap product by ZenAI.  
<img src="https://github.com/vuongminh1907/ComfyUI_ZenID/blob/main/examples/zenai.png" width="400" />

If you need a more polished and enhanced version, please contact us through:  
- 📱 **Facebook Page**: [ZenAI](https://web.facebook.com/zenai.vn)  
- ☎️ **Phone**: 0971912713 Miss. Chi  

🎉 Enjoy creating with ZenID! Let us know your feedback or suggestions.
