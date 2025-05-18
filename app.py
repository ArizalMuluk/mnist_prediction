from flask import Flask, request, jsonify, render_template
import torch as tc
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import ToTensor, Grayscale, Resize
import io
import base64

# Definisikan ulang kelas model CNN Anda
# Ini penting saat memuat state_dict, struktur model harus ada
class CNN(nn.Module):
  def __init__(self):
    super(CNN, self).__init__()

    self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
    self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
    self.conv2_drop = nn.Dropout2d()
    self.fc1 = nn.Linear(320, 50)
    self.fc2 = nn.Linear(50, 10)

  def forward(self, x):
    x = F.relu(F.max_pool2d(self.conv1(x), 2))
    x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
    x = x.view(-1, 320)
    x = F.relu(self.fc1(x))
    x = F.dropout(x, training=self.training)
    x = self.fc2(x)

    return F.softmax(x)

app = Flask(__name__)

# Muat model saat aplikasi Flask dimulai
device = tc.device('cuda' if tc.cuda.is_available() else 'cpu')
model = CNN().to(device)
model_save_path = './output/cnn_mnist.pt'

try:
    model.load_state_dict(tc.load(model_save_path, map_location=device))
    model.eval() # Set model ke mode evaluasi
    print("Model berhasil dimuat.")
except FileNotFoundError:
    print(f"Error: File model '{model_save_path}' tidak ditemukan. Pastikan model sudah disimpan di lokasi yang benar.")
    # Anda mungkin ingin menangani ini dengan lebih baik di aplikasi produksi
except Exception as e:
    print(f"Error saat memuat model: {e}")

@app.route('/')
def index():
    return render_template('index.html')

# Endpoint untuk prediksi
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', error="Tidak ada file gambar di request")

    file = request.files['file']

    if file.filename == '':
        return render_template('index.html', error="Tidak ada file gambar yang dipilih")

    if file:
        try:
            # Baca gambar dari stream file
            img_bytes = file.read()
            img = Image.open(io.BytesIO(img_bytes))

            # Preprocessing gambar agar sesuai dengan input model MNIST (grayscale, ukuran 28x28)
            transform = ToTensor()
            img_for_model = img.convert('L').resize((28, 28), Image.Resampling.LANCZOS)
            img_tensor = transform(img_for_model).unsqueeze(0).to(device)

            # Lakukan inferensi
            with tc.no_grad():
                output = model(img_tensor)
            prediction = output.argmax(dim=1, keepdim=True).item()

            # Konversi gambar asli ke base64 untuk ditampilkan
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            img_b64 = base64.b64encode(buffered.getvalue()).decode()
            image_data_url = f"data:image/png;base64,{img_b64}"

            return render_template(
                'index.html',
                image_data_url=image_data_url,
                predicted_digit=prediction,
                error=None
            )

        except Exception as e:
            return render_template('index.html', error=f"Gagal memproses gambar atau melakukan prediksi: {e}")

    return render_template('index.html', error="Terjadi kesalahan yang tidak diketahui")

if __name__ == '__main__':
    # Jalankan aplikasi Flask
    # Di lingkungan produksi, gunakan server WSGI seperti Gunicorn atau uWSGI
    app.run(debug=True) # debug=True hanya untuk pengembangan