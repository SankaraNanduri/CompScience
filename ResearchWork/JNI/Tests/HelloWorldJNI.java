public class HelloWorldJNI {
    // native method
    public native void sayHello();

    // Loading native library
    static {
        System.loadLibrary("HelloWorldLib");
    }

    public static void main(String[] args) {
        new HelloWorldJNI().sayHello();  // Call the native method
    }
}
