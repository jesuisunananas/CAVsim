import json
import folium
import random

def generate_map(json_file, output_html="detections_map.html"):
    print(f"📂 Loading data from {json_file}...")
    
    try:
        with open(json_file, 'r') as f:
            detections = json.load(f)
    except FileNotFoundError:
        print(f"❌ Error: Could not find {json_file}")
        return

    # Dictionary to group detections by their global ID
    tracks = {}
    all_lats = []
    all_lons = []

    for det in detections:
        obj_id = det.get('object_id')
        lat = det.get('gps_location', {}).get('latitude')
        lon = det.get('gps_location', {}).get('longitude')
        obj_type = det.get('object_type', 'unknown')

        # Skip entries that are missing critical data
        if not all([obj_id, lat, lon]):
            continue

        if obj_id not in tracks:
            # Assign a random hex color to each unique object for distinction
            tracks[obj_id] = {
                'type': obj_type,
                'coords': [],
                'color': "#{:06x}".format(random.randint(0, 0xBBBBBB)) # Kept darker for visibility on light maps
            }
        
        tracks[obj_id]['coords'].append((lat, lon))
        all_lats.append(lat)
        all_lons.append(lon)

    if not all_lats:
        print("❌ No valid GPS data found in the JSON file.")
        return

    # Calculate the geographic center to initialize the map
    center_lat = sum(all_lats) / len(all_lats)
    center_lon = sum(all_lons) / len(all_lons)

    print(f"🗺️  Found {len(tracks)} unique objects. Generating map...")
    
    # Initialize the map
    m = folium.Map(location=[center_lat, center_lon], zoom_start=19, max_zoom=22)

    # Draw the tracks on the map
    for obj_id, track_data in tracks.items():
        coords = track_data['coords']
        color = track_data['color']
        obj_type = track_data['type']

        # 1. Draw a connecting line (trajectory path) if the object moved over time
        if len(coords) > 1:
            folium.PolyLine(
                locations=coords,
                color=color,
                weight=3,
                opacity=0.7,
                tooltip=f"Path: {obj_id}"
            ).add_to(m)

        # 2. Draw tiny markers for historical positions (the breadcrumbs)
        for lat, lon in coords[:-1]:
            folium.CircleMarker(
                location=(lat, lon),
                radius=3,
                color=color,
                fill=True,
                fill_opacity=0.6,
                weight=1
            ).add_to(m)

        # 3. Draw a larger marker for the LAST known position
        last_lat, last_lon = coords[-1]
        folium.CircleMarker(
            location=(last_lat, last_lon),
            radius=7,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=1.0,
            weight=2,
            popup=f"<b>ID:</b> {obj_id}<br><b>Type:</b> {obj_type}<br><b>Detections:</b> {len(coords)}",
            tooltip=f"{obj_id} (Latest)"
        ).add_to(m)

    # Automatically pan and zoom the map to fit all the bounding boxes perfectly
    m.fit_bounds([[min(all_lats), min(all_lons)], [max(all_lats), max(all_lons)]])

    # Save the map to a local HTML file
    m.save(output_html)
    print(f"✅ Map successfully generated: {output_html}")
    print(f"🌐 Simply double-click '{output_html}' to open it in your web browser.")

if __name__ == "__main__":
    # Ensure this matches the output JSON from your MultiCameraPipeline
    generate_map('multi_cam_detections.json')