# ImageProcessing 影像處理工具
## 簡介    
ImageProcessing 類別提供了多種影像預處理技術，同時包含計算均方誤差 (MSE) 的方法，可用於辨識驗證碼。此類別被包裝成 main 函數，方便整合到您的應用程式中。    
---    

## 環境
```python
pip install -r requirements.txt
```

## 使用方法   
您可以使用以下程式碼將 ImageProcessing 類別整合到您的應用程式中：

```python
from ImageProcessing import ImageProcessing

# 指定輸入影像路徑
input_image_path="path/to/your/image.jpg"

# 如果希望保留中間影像，將 separate 參數設置為 True
ImageProcessing.main(input_image_path, separate=True)

```
---   
## 貢獻   
歡迎提供建議和改進！請隨時發送問題或拉取請求。    
