import 'package:dio/dio.dart';
import 'package:flutter/foundation.dart';
import 'package:latlong2/latlong.dart';
import 'airbnb_listing.dart';

class ApiService {
  final Dio _dio = Dio();

  String get baseUrl {
    if (kIsWeb) {
      return "http://localhost:8000"; // Web
    } else if (defaultTargetPlatform == TargetPlatform.android) {
      return "http://192.168.1.15:8000"; // Emulatore Android
    } else {
      return "http://192.168.1.15:8000"; // iOS o Desktop
    }
  }

  Future<List<AirbnbListing>> fetchListingsInArea(LatLng center, {double delta = 0.05}) async {
    try {
      final response = await _dio.get(
        '$baseUrl/distance/', 
        queryParameters: {
          'lat': center.latitude,
          'lon': center.longitude,
          'delta': delta,
        },
      );

      if (response.statusCode == 200) {
        final data = response.data;
        final List<dynamic> results = data['results'];

        return results.map((json) => AirbnbListing.fromJson(json)).toList();
      } else {
        throw Exception("Errore backend: ${response.statusCode}");
      }
    } catch (e) {
      print("Errore durante il fetch: $e");
      return [];
    }
  }
}