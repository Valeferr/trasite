import 'package:latlong2/latlong.dart';

class AirbnbListing {
  final String id;
  final String name;
  final LatLng location;
  final double trustScore; 
  final int realReviews;
  final int fakeReviews;
  final String hostName;
  final int price;

  AirbnbListing({
    required this.id,
    required this.name,
    required this.location,
    required this.trustScore,
    required this.realReviews,
    required this.fakeReviews,
    required this.hostName,
    required this.price,
  });

  factory AirbnbListing.fromJson(Map<String, dynamic> json) {
    final int legit = json['legit_count'] ?? 0;
    final int fake = json['fake_count'] ?? 0;
    final int totalReviews = legit + fake;

    double score = totalReviews > 0 ? legit / totalReviews : 0.0;

    return AirbnbListing(
      id: json['id_room']?.toString() ?? '0',
      name: json['name'] ?? 'Nome non disponibile',
      location: LatLng(
        (json['latitude'] as num).toDouble(), 
        (json['longitude'] as num).toDouble()
      ),
      trustScore: score,
      realReviews: legit,
      fakeReviews: fake,
      hostName: json['host_name'] ?? 'Host Sconosciuto',
      price: json['price'] ?? 0,
    );
  }
}