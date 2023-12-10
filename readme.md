### ImageProcessing 影像處理工具
簡介
ImageProcessing 類別提供了多種影像預處理技術，同時包含計算均方誤差 (MSE) 的方法，可用於辨識驗證碼。此類別被包裝成 main 函數，方便整合到您的應用程式中。

使用方法
您可以使用以下程式碼將 ImageProcessing 類別整合到您的應用程式中：

python
Copy code
from ImageProcessing import ImageProcessing

# 指定輸入影像路徑
input_image_path = "path/to/your/image.jpg"

# 如果希望保留中間影像，將 separate 參數設置為 True
ImageProcessing.main(input_image_path, separate=True)
歡迎根據您的特定需求修改和擴展此腳本。
