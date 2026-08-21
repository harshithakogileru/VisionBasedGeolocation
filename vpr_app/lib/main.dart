import 'dart:io';
import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:http/http.dart' as http;

void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      home: VPRScreen(),
    );
  }
}

class VPRScreen extends StatefulWidget {
  @override
  _VPRScreenState createState() => _VPRScreenState();
}

class _VPRScreenState extends State<VPRScreen> {
  File? _image;
  String latitude = "";
  String longitude = "";
  bool isLoading = false;

  final ImagePicker _picker = ImagePicker();

  // PICK IMAGE
  Future<void> pickImage() async {
    final pickedFile =
        await _picker.pickImage(source: ImageSource.gallery);

    if (pickedFile != null) {
      setState(() {
        _image = File(pickedFile.path);
        latitude = "";
        longitude = "";
      });
    }
  }

  // UPLOAD IMAGE TO FASTAPI
  Future<void> uploadImage() async {
    if (_image == null) return;

    setState(() {
      isLoading = true;
    });

    var request = http.MultipartRequest(
      'POST',
      Uri.parse("http://10.66.4.248:8000/predict"),
    );

    request.files.add(
      await http.MultipartFile.fromPath('file', _image!.path),
    );

    var response = await request.send();
    var responseData = await response.stream.bytesToString();

    var jsonData = json.decode(responseData);

    setState(() {
      latitude = jsonData["latitude"].toString();
      longitude = jsonData["longitude"].toString();
      isLoading = false;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.grey[200],
      body: SafeArea(
        child: SingleChildScrollView(
          child: Column(
            children: [

              SizedBox(height: 30),

              // TITLE
              Text(
                "Visual Place Recognition",
                textAlign: TextAlign.center,
                style: TextStyle(
                  fontSize: 28,
                  fontWeight: FontWeight.bold,
                  color: Colors.blue[700],
                ),
              ),

              SizedBox(height: 40),

              // BUTTONS
              Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [

                  ElevatedButton.icon(
                    style: ElevatedButton.styleFrom(
                      backgroundColor: Colors.blue,
                      padding: EdgeInsets.symmetric(
                          horizontal: 25, vertical: 20),
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(12),
                      ),
                    ),
                    onPressed: pickImage,
                    icon: Icon(Icons.photo, size: 28),
                    label: Text(
                      "Pick Image",
                      style: TextStyle(fontSize: 18),
                    ),
                  ),

                  SizedBox(width: 20),

                  ElevatedButton.icon(
                    style: ElevatedButton.styleFrom(
                      backgroundColor: Colors.green,
                      padding: EdgeInsets.symmetric(
                          horizontal: 25, vertical: 20),
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(12),
                      ),
                    ),
                    onPressed: uploadImage,
                    icon: Icon(Icons.upload, size: 28),
                    label: Text(
                      "Upload Photo",
                      style: TextStyle(fontSize: 18),
                    ),
                  ),
                ],
              ),

              SizedBox(height: 30),

              // LOADING
              if (isLoading)
                CircularProgressIndicator(),

              SizedBox(height: 20),

              // IMAGE PREVIEW
              if (_image != null)
                Padding(
                  padding: EdgeInsets.all(20),
                  child: ClipRRect(
                    borderRadius: BorderRadius.circular(15),
                    child: Image.file(
                      _image!,
                      height: 200,
                    ),
                  ),
                ),

              SizedBox(height: 20),

              // LOCATION CARD
              if (latitude.isNotEmpty && longitude.isNotEmpty)
                Container(
                  margin: EdgeInsets.symmetric(horizontal: 20),
                  padding: EdgeInsets.all(20),
                  decoration: BoxDecoration(
                    color: Colors.blueGrey[800],
                    borderRadius: BorderRadius.circular(20),
                  ),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [

                      Row(
                        children: [
                          Icon(Icons.location_on,
                              color: Colors.red),
                          SizedBox(width: 8),
                          Text(
                            "Detected Location:",
                            style: TextStyle(
                              color: Colors.white,
                              fontSize: 20,
                              fontWeight: FontWeight.bold,
                            ),
                          ),
                        ],
                      ),

                      SizedBox(height: 20),

                      Row(
                        children: [
                          Icon(Icons.my_location,
                              color: Colors.white),
                          SizedBox(width: 10),
                          Expanded(
                            child: Text(
                              "Latitude: $latitude",
                              style: TextStyle(
                                  color: Colors.white,
                                  fontSize: 18),
                            ),
                          ),
                        ],
                      ),

                      SizedBox(height: 10),

                      Row(
                        children: [
                          Icon(Icons.public,
                              color: Colors.white),
                          SizedBox(width: 10),
                          Expanded(
                            child: Text(
                              "Longitude: $longitude",
                              style: TextStyle(
                                  color: Colors.white,
                                  fontSize: 18),
                            ),
                          ),
                        ],
                      ),
                    ],
                  ),
                ),

              SizedBox(height: 40),
            ],
          ),
        ),
      ),
    );
  }
}
