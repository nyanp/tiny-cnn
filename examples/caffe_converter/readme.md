# Import Caffe Model to tiny-dnn
tiny-dnn can import Caffe's trained models and store them in a binary, architecture independant way (32bit/64bit compatible) for easy reuse on linux, mac, iOS and Android.

## Prerequisites for this example
- Google protobuf
- OpenCV

## Build

1 Use ```protoc``` to generte caffe.pb.cc and caffe.pb.h.
```bash
cd tiny_dnn/io/caffe
protoc caffe.proto --cpp_out=./
```

2 Compile ```tiny_dnn/io/caffe/caffe.pb.cc``` and ```examples/caffe_converter/caffe_converter.cpp``` and link them.

## Usage
```bash
./caffe_converter.bin [mode] [mode-file] [model-file] [trained-file] [mean-file] [label-file] 
```
Where **mode** can be either test, archive or archive-test

### Testing a Pre-trained Model
In the [pre-trained CaffeNet](https://github.com/BVLC/caffe/tree/master/examples/cpp_classification) model,
```
./caffe_converter.bin\
 test\
 cat.jpg\
 deploy.prototxt\
 bvlc_reference_caffenet.caffemodel\
 imagenet_mean.binaryproto\
 synset_words.txt
```

### Archiving a Pre-trained Model
In the [pre-trained CaffeNet](https://github.com/BVLC/caffe/tree/master/examples/cpp_classification) model,
```
./caffe_converter.bin\
 archive\
 bvlc_reference_caffenet.tinydnn\
 deploy.prototxt\
 bvlc_reference_caffenet.caffemodel\
 imagenet_mean.binaryproto\
 synset_words.txt
```
This creates the archived version of the network bvlc_reference_caffenet.tinydnn.  The archived file contains the converted tinydnn network, the mean file and the labels.  

### Testing an Archived Model
You can test an archived version of the network by running,
```
./caffe_converter.bin\
 archive-test\
 bvlc_reference_caffenet.tinydnn\
 cat.jpg
```

It can be used at a later time on your platform of choice (without caffe/protobuf dependencies) by doing something like, 

```
#include "tiny_dnn/models/archive.h"
using namespace tiny_dnn;
using namespace std;

...

models::archive archive("/path/to/model_file");
image<float> img("/path/to/image_file", image_type::bgr);
auto predictions(archive.predict(img));
        
for (auto prediction : predictions) {
    cout << prediction.label << "," << prediction.confidence << endl;
}

// take a look at caffe_converter.cpp for more details
```
## Restrictions
- tiny-dnn's converter only supports single input/single output network without branch.
