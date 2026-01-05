import 'package:flutter/material.dart';
import 'package:flutter_map/flutter_map.dart';
import 'package:flutter_map_marker_cluster/flutter_map_marker_cluster.dart';
import 'package:latlong2/latlong.dart';
import 'package:fl_chart/fl_chart.dart';
import 'airbnb_listing.dart';
import 'api_service.dart'; 

class AirbnbMapScreen extends StatefulWidget {
  const AirbnbMapScreen({super.key});

  @override
  State<AirbnbMapScreen> createState() => _AirbnbMapScreenState();
}

class _AirbnbMapScreenState extends State<AirbnbMapScreen> {
  final MapController _mapController = MapController();
  final ApiService _apiService = ApiService();
  
  List<AirbnbListing> listings = [];
  bool isLoading = true;

  final LatLng initialCenter = const LatLng(40.8518, 14.2681);

  @override
  void initState() {
    super.initState();
    _loadListings();
  }

  Future<void> _loadListings() async {
    setState(() => isLoading = true);
    final data = await _apiService.fetchListingsInArea(initialCenter, delta: 0.1);
    
    setState(() {
      listings = data;
      isLoading = false;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text("Analisi Airbnb"),
        actions: [
          IconButton(
            icon: const Icon(Icons.refresh),
            onPressed: _loadListings,
          )
        ],
      ),
      body: Stack(
        children: [
          FlutterMap(
            mapController: _mapController,
            options: MapOptions(
              initialCenter: initialCenter,
              initialZoom: 13.0,
              maxZoom: 18.0,
            ),
            children: [
              TileLayer(
                urlTemplate: 'https://tile.openstreetmap.org/{z}/{x}/{y}.png',
                userAgentPackageName: 'com.trasite.app',
              ),
              MarkerClusterLayerWidget(
                options: MarkerClusterLayerOptions(
                  maxClusterRadius: 120,
                  size: const Size(40, 40),
                  alignment: Alignment.center,
                  padding: const EdgeInsets.all(50),
                  builder: (context, markers) {
                    return Container(
                      decoration: BoxDecoration(
                        color: Colors.blue,
                        shape: BoxShape.circle,
                        border: Border.all(color: Colors.white, width: 2),
                      ),
                      child: Center(
                        child: Text(
                          markers.length.toString(),
                          style: const TextStyle(color: Colors.white, fontWeight: FontWeight.bold),
                        ),
                      ),
                    );
                  },
                  markers: listings.map((listing) {
                    return Marker(
                      point: listing.location,
                      width: 40,
                      height: 40,
                      child: GestureDetector(
                        onTap: () => _showListingDetails(context, listing),
                        child: Icon(
                          Icons.location_on,
                          size: 40,
                          color: listing.trustScore > 0.9 ? Colors.green[800] :
                                 listing.trustScore > 0.7 ? Colors.green :
                                 listing.trustScore > 0.4 ? Colors.orange : Colors.red,
                          shadows: const [
                            Shadow(blurRadius: 10, color: Colors.black26, offset: Offset(2, 2))
                          ],
                        ),
                      ),
                    );
                  }).toList(),
                ),
              ),
            ],
          ),
          
          if (isLoading)
            const Center(
              child: Card(
                child: Padding(
                  padding: EdgeInsets.all(20.0),
                  child: CircularProgressIndicator(),
                ),
              ),
            ),
        ],
      ),
    );
  }

  void _showListingDetails(BuildContext context, AirbnbListing listing) {
    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.transparent,
      builder: (context) {
        return DraggableScrollableSheet(
          initialChildSize: 0.5,
          minChildSize: 0.3,
          maxChildSize: 0.9,
          builder: (context, scrollController) {
            return Container(
              decoration: const BoxDecoration(
                color: Colors.white,
                borderRadius: BorderRadius.vertical(top: Radius.circular(20)),
              ),
              padding: const EdgeInsets.all(20),
              child: SingleChildScrollView(
                controller: scrollController,
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Center(
                      child: Container(
                        width: 40, height: 5, 
                        decoration: BoxDecoration(color: Colors.grey[300], borderRadius: BorderRadius.circular(10))
                      ),
                    ),
                    const SizedBox(height: 20),
                    
                    Text(listing.name, style: const TextStyle(fontSize: 22, fontWeight: FontWeight.bold)),
                    Text("Host: ${listing.hostName}", style: TextStyle(color: Colors.grey[600])),
                    Text("Prezzo: €${listing.price}", style: const TextStyle(fontSize: 18, fontWeight: FontWeight.w500)),
                    
                    const Divider(height: 30),

                    const Text("Grado di Veridicità", style: TextStyle(fontSize: 16, color: Colors.grey)),
                    const SizedBox(height: 10),
                    LinearProgressIndicator(
                      value: listing.trustScore,
                      minHeight: 10,
                      backgroundColor: Colors.grey[300],
                      color: listing.trustScore > 0.7 ? Colors.green : Colors.red,
                      borderRadius: BorderRadius.circular(5),
                    ),
                    const SizedBox(height: 5),
                    Text("${(listing.trustScore * 100).toInt()}% Affidabile", 
                         style: const TextStyle(fontWeight: FontWeight.bold)),

                    const SizedBox(height: 30),

                    const Text("Analisi Recensioni", style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold)),
                    const SizedBox(height: 20),
                    SizedBox(
                      height: 200,
                      child: PieChart(
                        PieChartData(
                          sections: [
                            PieChartSectionData(
                              value: listing.realReviews.toDouble(),
                              color: Colors.green,
                              title: '${listing.realReviews}',
                              radius: 60,
                              titleStyle: const TextStyle(color: Colors.white, fontWeight: FontWeight.bold),
                            ),
                            PieChartSectionData(
                              value: listing.fakeReviews.toDouble(),
                              color: Colors.redAccent,
                              title: '${listing.fakeReviews}',
                              radius: 50,
                              titleStyle: const TextStyle(color: Colors.white, fontWeight: FontWeight.bold),
                            ),
                          ],
                          sectionsSpace: 2,
                          centerSpaceRadius: 40,
                        ),
                      ),
                    ),
                    
                    const SizedBox(height: 10),
                    Center(
                      child: Row(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          _buildLegend(Colors.green, "Vere"),
                          const SizedBox(width: 20),
                          _buildLegend(Colors.redAccent, "Fake"),
                        ],
                      ),
                    ),
                  ],
                ),
              ),
            );
          },
        );
      },
    );
  }

  Widget _buildLegend(Color color, String text) {
    return Row(
      children: [
        Container(width: 16, height: 16, color: color),
        const SizedBox(width: 5),
        Text(text),
      ],
    );
  }
}