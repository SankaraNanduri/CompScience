#include <jni.h>
#include <iostream>
#include "HelloWorldJNI.h"

// Implementation of the native method
extern "C" JNIEXPORT void JNICALL HelloWorldJNI_sayHello(JNIEnv* env, jobject thisObj) {
    std::cout << "Hello, World from C++!" << std::endl;
}
