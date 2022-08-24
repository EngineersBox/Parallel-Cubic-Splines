kernel void scaleRow(double** inMatrix, uint rows, uint cols, float** outMatrix, int currentCol) {
    int threadX = get_global_id(0);
    int threadY = get_global_id(1);
    if (currentCol == threadY && threadY < rows && threadX < cols) {
        outMatrix[threadY * cols][threadX] = inMatrix[threadY * cols][threadX] / inMatrix[currentCol * cols][currentCol];
    }
}

kernel void subtractRow(double** inMatrix, uint rows, uint cols, double** outMatrix, int currentCol) {
    int threadX = get_global_id(0);
    int threadY = get_global_id(1);
    if (currentCol != threadY && threadY < rows && threadX < cols) {
        outMatrix[threadY * cols][threadX] = inMatrix[threadY * cols][threadX] - (inMatrix[currentCol * cols][ threadX] * inMatrix[threadY * cols][currentCol]);
    }
}