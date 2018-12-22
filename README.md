# Running retrained Mobile Net in dnn EmguCV/OpenCV

OpenCV provide a package called dnn that allows us to implement deep neural network in our application. In this tutorial, I will walk you step by step to implement a retrained Mobile Net to your C# application with EmguCV (.Net wrapper for OpenCV).

# Appetizer (pre-requisite)

 1. Retrained Mobile Net model
 
    Mobile Net is a small efficient convolutional neural network provided by Google. There is [a very easy-to-go turotial to retrain a Mobile Net model to classify images of flowers](https://codelabs.developers.google.com/codelabs/tensorflow-for-poets/#0). It's a very good way to make a neural network model for classification even if you've never worked with a tensorflow model before.
 2. Tensorflow 
 
    As we're working with Tensorflow model, we need to have tensorflow in our machines. You should install it from source because we will need `graph_transforms` tool later. If you install it using `pip`, it's ok but you still need to build `graph_transforms` separately. All about installing Tensorflow is available in [the official Tensorflow website](https://www.tensorflow.org/install/).
 3. EmguCV
 
    Create a C# project and add EmguCV using Nuget Package Management.
    
# Main course

 1. Make some changes in the Tensorflow model so it can be used in EmguCV.dnn. 
 
    In this part, you can do it either in Windows or Ubuntu. I recommend to do it in Ubuntu because it's simpler and less buggy. Suppose we have a mobile net with input image resolution of 128x128 `retrained_mobilenet_1.0_128.pb`.
    
    1.1. Optimize_for_inference
    
    For Windows
        
    ```
    python -m tensorflow.python.tools.optimize_for_inference.py \
     --input retrained_mobilenet_1.0_128.pb \
     --output opt_graph.pb \
     --frozen_graph True \
     --input_names input \
     --output_names final_result
    ``` 
    
    For Ubuntu
    
    ```
    python ~/tensorflow/tensorflow/python/tools/optimize_for_inference.py \
     --input retrained_mobilenet_1.0_128.pb \
     --output opt_graph.pb \
     --frozen_graph True \
     --input_names input \
     --output_names final_result
    ```
    
    1.2. Graph_transforms
    
    ```
    tensorflow/bazel-bin/tensorflow/tools/graph_transforms/transform_graph \
     --in_graph=opt_graph.pb \
     --out_graph=mobilenet_for_dnn.pb \
     --inputs=input \
     --outputs=final_result \
     --transforms="fold_constants remove_nodes(op=Squeeze) remove_nodes(op=PlaceholderWithDefault) strip_unused_nodes(type=float, shape=\"1,128,128,3\") sort_by_execution_order "
    ```
    
    Remember to correct the `shape` parameter as your model's input resolution. Here is `1,128,128,3`.
    
 2. Use the model in EmguCV
 
    Using the Dnn
 
     ```
     using Emgu.CV.Dnn;
     ```
 
    Read the model
      ```
      mobile_net = DnnInvoke.ReadNetFromTensorflow("mobilenet_for_dnn.pb");
      ```
  
    Read the labels
      ```
      List<string> labels = new List<string>();
      using (System.IO.StreamReader fr = new System.IO.StreamReader("labels.txt"))
      {
          string line = "";
          while ((line = fr.ReadLine()) != null)
          {
              labels.Add(line);
          }
      }
      //"label.txt" contains labels for all classes in the model.
      ```
      
    Read image 
 
    ```
    Mat m = new Mat("image_to_classify.jpg");
    Mat blob = DnnInvoke.BlobFromImage(m, 1, new Size(128, 128));
    ```
 
    Set input and forward the image blob
 
    ```
    tensor_net.SetInput(blob, "input");
    Mat detection = tensor_net.Forward("final_result");
    ```
 
    Get the output 
 
     ```
     byte[] data = new byte[116];
     detection.CopyTo(data);

     List<float> confidence = new List<float>();

     for (int i = 0; i < data.Length / 4; i++)
     {
         confidence.Add(BitConverter.ToSingle(data, i * 4));

     }
     //The output is a byte array containing the confidence of every class in every 4 bytes. 
     //My model is to classify 29 objects, so the output has 4x29=116 bytes.
     //The BitConverter.ToSingle(data, i * 4) will convert 4 bytes to a float value.
     ```
    
    Get the prediction
    
    ```
    int maxIndex = confidence.IndexOf(confidence.Max());
    float sum = confidence.Sum(); // make sure the sum is 1.0
    string prediction = labels[maxIndex];
    ```
    
# Dessert (result)

