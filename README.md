# Mid-term-Project-TTKH
# Optimization Algorithms Comparison: Adam vs. Genetic Algorithm

Dự án này thực hiện mô phỏng và so sánh hiệu năng giữa thuật toán tối ưu dựa trên Gradient (**Adam Optimizer**) và giải thuật di truyền (**Genetic Algorithm - GA**) trên hai hàm mục tiêu kinh điển: **Rosenbrock** (đơn cực trị) và **Rastrigin** (đa cực trị).

## 📋 Mục lục
- [Giới thiệu](#giới-thiệu)
- [Cài đặt](#cài-đặt)
- [Kết quả thực nghiệm](#kết-quả-thực-nghiệm)
- [Hình ảnh minh họa](#hình-ảnh-minh-họa)

## 📖 Giới thiệu

- **Mục tiêu:** Tìm cực tiểu toàn cục (Global Minima) của hàm số.
- **Input:** Điểm xuất phát chung $x_0 = [-1.0, 2.0]$.
- **Hàm mục tiêu:**
  1. **Rosenbrock:** Thung lũng hẹp, hình Parabol. Global Min tại $[1, 1]$.
  2. **Rastrigin:** Bề mặt gồ ghề, nhiều cực tiểu địa phương. Global Min tại $[0, 0]$.

## ⚙️ Cài đặt

Yêu cầu môi trường Python 3.x và các thư viện:

```bash
pip install numpy matplotlib
