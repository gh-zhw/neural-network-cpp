#pragma once

#include <vector>

class Matrix
{
public:
    Matrix() = default;
    Matrix(size_t r, size_t c, float fill = 0.0f) : row_(r), col_(c) {
        data.resize(r, std::vector<float>(c, fill));
    }
    Matrix(size_t, size_t, std::vector<float>);
    Matrix(const Matrix&) = default;
    Matrix& operator=(const Matrix&) = default;
    Matrix(Matrix&&) noexcept;
    Matrix& operator=(Matrix&&) noexcept;

    float get(size_t i, size_t j) const { return data[i][j]; }
    void set(size_t, size_t, float);
    void assign(float);
    size_t row() const { return row_; }
    size_t col() const { return col_; }

    // matrix math operation
    Matrix T() const;

    Matrix add(const Matrix&) const;
    Matrix sub(const Matrix&) const;
    Matrix matMul(const Matrix&) const;
    Matrix elementMul(const Matrix&) const;

    Matrix add(float) const;
    Matrix sub(float) const;
    Matrix elementMul(float) const;

    Matrix operator+(const Matrix&) const;
    Matrix operator-(const Matrix&) const;
    Matrix operator*(const Matrix&) const;

    Matrix operator+(float) const;
    Matrix operator-(float) const;
    Matrix operator*(float) const;
    Matrix operator/(float) const;

    friend Matrix operator+(float, const Matrix&);
    friend Matrix operator*(float, const Matrix&);

private:
    size_t row_ = 0, col_ = 0;
    std::vector<std::vector<float>> data;
};
