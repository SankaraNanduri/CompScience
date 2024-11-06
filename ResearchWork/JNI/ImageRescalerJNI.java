public class ImageRescalerJNI {

    // Native method declaration
    public native void rescaleImage(short[] inputImage, byte[] outputImage, int innerBound, int upperBound, int size);

    // Load the native library
    static {
        System.loadLibrary("ImageRescalerLib");
    }

    public static void main(String[] args) {
        int size = 4000000;
        short[] inputImage = new short[size];
        byte[] outputImage = new byte[size];
        int innerBound = 1000;
        int upperBound = 30000;

        // Fill inputImage with random values
        for (int i = 0; i < size; i++) {
            inputImage[i] = (short)(innerBound + (Math.random() * (upperBound - innerBound)));
        }

        // Create an instance of the class and call the native method
        ImageRescalerJNI rescaler = new ImageRescalerJNI();
        rescaler.rescaleImage(inputImage, outputImage, innerBound, upperBound, size);

        System.out.println("Image rescaling completed.");
    }
}
