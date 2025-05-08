import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import load_model
from glob import glob
import subprocess
import sys
from PIL import Image, ImageEnhance
import matplotlib.dates as mdates


def run_yolo_with_logs():
    # Create a pipe to capture the output
    process = subprocess.Popen(
        [sys.executable, "detect_v2.py"],  # Run the YOLO detection script
        cwd="yolo_pattern",  # run inside the yolo_pattern folder
        stdout=subprocess.PIPE,  # Capture standard output
        stderr=subprocess.PIPE,  # Capture standard error
        text=True  # Set the output as text
    )

    # Read the output line by line and display it in real-time
    for line in process.stdout:
        if line:
            st.text(line)  # Stream output to Streamlit app

    # Also capture stderr if there's any error
    for line in process.stderr:
        if line:
            st.text(line)  # Stream errors to Streamlit app

    process.wait()  # Wait for the process to finish

st.set_page_config(layout="wide")
st.title("📊 Stock Prediction & 🧠 Pattern Detection App")

tab1, tab2, tab3, tab4 = st.tabs(["📈 Time Series Prediction", "🔍 Direction Classification (LSTM/GRU)", "📷 YOLO Pattern Detection", "📁 View YOLO Results"])

# ---------- Tab 1 ----------
with tab1:
    model_type = st.selectbox("Choose Model Type", ["GRU", "LSTM"])
    model_folder = f"models/{model_type}"

    # Fetch available models based on selected type
    available_models = [os.path.basename(path) for path in glob(f"{model_folder}/*{model_type}*.keras")]

    # Select model file based on selected type (e.g., GRU_xxx_model.keras)
    selected_model = st.selectbox("Choose Model File", available_models)

    # Determine the corresponding data file based on the selected model
    selected_ticker = selected_model.replace(f"{model_type}_", "").replace("_model.keras", "")
    selected_data_file = f"data/{selected_ticker}.csv"

    if st.button("Run Prediction"):
        st.info(f"Loading {selected_model}...")

        df = pd.read_csv(selected_data_file, index_col=0, parse_dates=True)
        df.drop('ticker', axis=1, inplace=True)

        scaler_close = MinMaxScaler()
        df['close_scaled'] = scaler_close.fit_transform(df[['close']])

        def create_sequences(data, lookback=30):
            X, y, idx = [], [], data.index.to_list()
            for i in range(len(data) - lookback):
                X.append(data.iloc[i:i + lookback].values)
                y.append(data.iloc[i + lookback, 0])
            return np.array(X), np.array(y), idx[lookback:]

        lookback = 30
        X, y, index_list = create_sequences(df[['close_scaled']], lookback)
        index_list = df.index[lookback:]
        test_mask = index_list >= '2024-09-01'
        X_test, y_test = X[test_mask], y[test_mask]
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

        model = load_model(os.path.join(model_folder, selected_model))
        y_pred = model.predict(X_test)

        y_pred_rescaled = scaler_close.inverse_transform(y_pred.reshape(-1, 1))
        y_test_rescaled = scaler_close.inverse_transform(y_test.reshape(-1, 1))


        # Plot
        fig, ax = plt.subplots(figsize=(8,4))
        ax.plot(y_test_rescaled[:-7], label="Actual", color="blue")
        ax.plot(y_pred_rescaled, label="Predicted", color="red")

        ax.get_xaxis().set_visible(False)
        ax.set_ylabel("Close Price")
        ax.legend()
        st.pyplot(fig, use_container_width=False)

        # Metrics
        mae = mean_absolute_error(y_test_rescaled, y_pred_rescaled)
        mse = mean_squared_error(y_test_rescaled, y_pred_rescaled)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test_rescaled, y_pred_rescaled)
        st.markdown(f"""
        **Metrics**:
        - MAE: `{mae:.4f}`
        - MSE: `{mse:.4f}`
        - RMSE: `{rmse:.4f}`
        - R² Score: `{r2:.4f}`
        """)

        # Error Distribution
        errors = y_test_rescaled - y_pred_rescaled
        fig2, ax2 = plt.subplots(figsize=(6,3))
        ax2.hist(errors, bins=20, edgecolor='black')
        ax2.set_title("Error Distribution")
        st.pyplot(fig2, use_container_width=False)

# ---------- Tab 2 ----------
import streamlit as st
import os
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt

with tab2:
    st.header("🔍 Direction Classification")

    # Chọn loại mô hình (LSTM hoặc GRU)
    model_type = st.selectbox("Choose Model Type", ["LSTM", "GRU"], key="tab5_model_type")
    model_folder = f"models/{model_type}"

    # Lấy danh sách các mô hình phân loại có sẵn
    available_models = [os.path.basename(path) for path in glob(f"{model_folder}/{model_type}_*_Classification.pth")]
    selected_model = st.selectbox("Choose Model File", available_models, key="tab5_model_select")

    # Xác định ticker và file dữ liệu tương ứng
    selected_ticker = selected_model.replace(f"{model_type}_", "").replace("_Classification.pth", "")
    selected_data_file = f"data/{selected_ticker}_PHARM_VNINDEX.csv"

    if st.button("Run Classification", key="tab5_run"):
        st.info(f"Loading {selected_model} for classification...")


        # Hàm load và xử lý dữ liệu
        def load_and_preprocess_data(file_path, ticker, sequence_length=12):
            df = pd.read_csv(file_path, index_col='time')
            df.index = pd.to_datetime(df.index)

            split_date = pd.to_datetime('2025-01-01')
            test_df = df[df.index >= split_date]

            if ticker not in test_df.columns:
                st.error(f"Ticker {ticker} không tồn tại trong dữ liệu. Các cột có sẵn: {list(test_df.columns)}")
                st.stop()

            features_test = test_df[[ticker, 'VNINDEX_lag2', 'PHARMA_lag1']].values
            target_test = test_df['target'].values
            test_dates = test_df.index

            scaler = StandardScaler()
            features_test = scaler.fit_transform(features_test)

            X_test, y_test, test_dates_seq = [], [], []
            for i in range(len(features_test) - sequence_length):
                X_test.append(features_test[i:i + sequence_length])
                y_test.append(target_test[i + sequence_length])
                test_dates_seq.append(test_dates[i + sequence_length])
            X_test = np.array(X_test)
            y_test = np.array(y_test)
            test_dates_seq = pd.to_datetime(test_dates_seq)

            X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
            y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

            test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
            test_loader = DataLoader(test_dataset, batch_size=32)

            return test_loader, X_test, y_test, test_dates_seq


        # Định nghĩa mô hình LSTM/GRU
        class LSTMModel(nn.Module):
            def __init__(self, input_dim, hidden_dim, output_dim):
                super(LSTMModel, self).__init__()
                self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
                self.fc = nn.Linear(hidden_dim, output_dim)

            def forward(self, x):
                lstm_out, _ = self.lstm(x)
                out = self.fc(lstm_out[:, -1, :])
                return out


        class GRUModel(nn.Module):
            def __init__(self, input_dim, hidden_dim, output_dim):
                super(GRUModel, self).__init__()
                self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
                self.fc = nn.Linear(hidden_dim, output_dim)

            def forward(self, x):
                gru_out, _ = self.gru(x)
                out = self.fc(gru_out[:, -1, :])
                return out


        # Hàm load mô hình
        def load_model(model_type, input_dim, hidden_dim, output_dim, path):
            if model_type == "LSTM":
                model = LSTMModel(input_dim, hidden_dim, output_dim)
            else:
                model = GRUModel(input_dim, hidden_dim, output_dim)
            if os.path.exists(path):
                model.load_state_dict(torch.load(path))
                st.success(f"Model loaded from {path}")
            else:
                st.error(f"No saved model found at {path}")
                return None
            return model


        # Hàm đánh giá mô hình (đã sửa để ánh xạ threshold)
        def evaluate(model, loader, model_type, ticker):
            # Ánh xạ threshold dựa trên model_type và ticker
            threshold_map = {
                ('DHG', 'LSTM'): 0.5,
                ('DHG', 'GRU'): 0.5,
                ('IMP', 'LSTM'): 0.45,
                ('IMP', 'GRU'): 0.45,
                ('TRA', 'LSTM'): 0.5,
                ('TRA', 'GRU'): 0.6
            }
            threshold = threshold_map.get((ticker, model_type))  # Mặc định 0.5 nếu không tìm thấy

            model.eval()
            predictions, labels = [], []
            with torch.no_grad():
                for batch_x, batch_y in loader:
                    outputs = model(batch_x)
                    predictions.extend(outputs.squeeze().numpy())
                    labels.extend(batch_y.squeeze().numpy())

            predictions_prob = torch.sigmoid(torch.tensor(predictions)).numpy()
            labels = np.array(labels)

            # Sử dụng threshold tương ứng để phân loại
            predictions = predictions_prob > threshold
            accuracy = (predictions == labels).mean()
            precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')

            return predictions, labels, accuracy, precision, recall, f1, threshold


        # Hàm hiển thị bảng kết quả
        def display_table(predictions, labels, dates, max_rows=100):
            df = pd.DataFrame({
                'Date': dates[:max_rows],
                'Actual': labels[:max_rows],
                'Predicted': predictions[:max_rows]
            })
            return df


        # Hàm vẽ biểu đồ
        def plot_predictions(predictions, labels, dates, max_points=100):
            fig, ax = plt.subplots(figsize=(12, 6))
            dates_subset = dates[:max_points]
            labels_subset = labels[:max_points]
            predictions_subset = predictions[:max_points]

            jitter = 0.02
            labels_jittered = labels_subset + np.random.uniform(-jitter, jitter, size=len(labels_subset))
            predictions_jittered = predictions_subset + np.random.uniform(-jitter, jitter, size=len(predictions_subset))

            ax.scatter(dates_subset, labels_jittered, label='Actual', color='blue', marker='o', alpha=0.5, s=50)
            ax.scatter(dates_subset, predictions_jittered, label='Predicted', color='red', marker='x', alpha=0.5, s=50)

            ax.set_title(f'Actual vs Predicted Trends for {selected_ticker} ({model_type})')
            ax.set_xlabel('Date')
            ax.set_ylabel('Trend (1: Up, 0: Down)')
            ax.set_ylim(-0.1, 1.1)
            ax.set_yticks([0, 1], ['0', '1'])
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.tick_params(axis='x', rotation=45)
            plt.tight_layout()
            return fig


        # Load và xử lý dữ liệu với ticker được truyền vào
        try:
            test_loader, X_test, y_test, test_dates = load_and_preprocess_data(selected_data_file, selected_ticker,
                                                                               sequence_length=12)
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            st.stop()

        # Xác định các tham số mô hình
        input_dim = X_test.shape[2]  # Số đặc trưng (ticker, VNINDEX_lag2, PHARMA_lag1) = 3
        hidden_dim = 128  # Phải khớp với giá trị khi huấn luyện
        output_dim = 1  # Dự đoán nhị phân (tăng/giảm)

        # Load mô hình
        model_path = os.path.join(model_folder, selected_model)
        model = load_model(model_type, input_dim, hidden_dim, output_dim, model_path)
        if model is None:
            st.stop()

        # Đánh giá và lấy dự đoán (truyền model_type và ticker để ánh xạ threshold)
        predictions, labels, accuracy, precision, recall, f1, threshold = evaluate(model, test_loader, model_type,
                                                                                   selected_ticker)

        # Hiển thị chỉ số đánh giá
        st.markdown(f"""
            **Metrics**:
            - Accuracy: `{accuracy:.4f}`
            - Precision: `{precision:.4f}`
            - Recall: `{recall:.4f}`
            - F1-Score: `{f1:.4f}`
            - Threshold: `{threshold:.2f}`
            """)

        # Hiển thị bảng kết quả
        st.subheader("Sample of Actual vs Predicted Values")
        results_df = display_table(predictions, labels, test_dates, max_rows=100)
        st.dataframe(results_df)

        # Vẽ biểu đồ
        st.subheader("Actual vs Predicted Trends")
        fig = plot_predictions(predictions, labels, test_dates, max_points=100)
        st.pyplot(fig, use_container_width=False)

# ---------- Tab 3 ----------
with tab3:
    if "yolo_started" not in st.session_state:
        st.session_state.yolo_started = False
        st.session_state.yolo_logs = ""

    if not st.session_state.yolo_started:
        st.warning("This runs a live CV script. Ensure your YOLO weights and CV environment are ready.")

    if st.button("🔄 Reset YOLO Tab"):
        st.session_state.yolo_started = False
        st.session_state.yolo_logs = ""
        st.rerun()

    if st.button("▶️ Run YOLO Pattern Detection"):
        st.success("▶️ Running YOLO CV script...")
        run_yolo_with_logs()  # This will display logs directly in Streamlit
with tab4:
    base_dir = "yolo_pattern/runs/detect"
    st.header("🖼️ View YOLO Predictions")
    # Step 1: List available session folders (e.g., session1, session2)
    session_dirs = sorted([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))])

    selected_session = st.selectbox("Select Session Folder", session_dirs)

    if selected_session:
        session_path = os.path.join(base_dir, selected_session)

        # Step 2: Get all predict* folders inside selected session
        predict_dirs = sorted(
            [d for d in os.listdir(session_path) if
             d.startswith("predict") and os.path.isdir(os.path.join(session_path, d))],
            key=lambda x: os.path.getmtime(os.path.join(session_path, x)),
            reverse=True
        )

        selected_predict_folder = st.selectbox("Select Predict Folder", predict_dirs)

        if selected_predict_folder:
            predict_path = os.path.join(session_path, selected_predict_folder)

            # Step 3: Load images inside the predict folder
            image_paths = glob(os.path.join(predict_path, "*.jpg")) + \
                          glob(os.path.join(predict_path, "*.png"))

            if image_paths:
                selected_image = st.selectbox("Choose an image to view", image_paths)

                if selected_image:
                    image = Image.open(selected_image)

                    st.image(image, caption=os.path.basename(selected_image),width=1000)
            else:
                st.warning("No images found in this folder.")


