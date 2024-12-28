plugins {
    alias(libs.plugins.android.application)
}

android {
    namespace = "com.example.fashion_image_retrieval"
    compileSdk = 34

    defaultConfig {
        applicationId = "com.example.fashion_image_retrieval"
        minSdk = 24
        targetSdk = 34
        versionCode = 1
        versionName = "1.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }
    buildFeatures {
        viewBinding = true
        dataBinding = true
    }
}

dependencies {

    implementation(libs.appcompat)
    implementation(libs.material)
    implementation(libs.constraintlayout)
    implementation(libs.lifecycle.livedata.ktx)
    implementation(libs.lifecycle.viewmodel.ktx)
    implementation(libs.navigation.fragment)
    implementation(libs.navigation.ui)
    testImplementation(libs.junit)
    androidTestImplementation(libs.ext.junit)
    androidTestImplementation(libs.espresso.core)
    implementation("com.github.rachelxj:DropdownMenu:v1.0.0")
    implementation("com.github.bumptech.glide:glide:4.15.1")
    implementation("com.squareup.okhttp3:okhttp:4.10.0")
    annotationProcessor("com.github.bumptech.glide:compiler:4.15.1")
}