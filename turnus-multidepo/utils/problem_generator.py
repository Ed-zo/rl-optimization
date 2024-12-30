import random
import math
import os

def distance_calculate(x1: int, y1: int, x2: int, y2: int) -> int:
    """Calculate Euclidean distance between two points."""
    x = x1 - x2
    y = y1 - y2
    sqr_dist = x**2 + y**2
    dist = math.ceil(math.sqrt(sqr_dist))
    
    if sqr_dist < 0:
        print(f"wrong distance {sqr_dist}")
    
    return dist

def u_integer(min_val: int, max_val: int) -> int:
    """Generate a random integer between min and max."""
    return math.ceil(min_val + random.random() * (max_val - min_val))

def generate_route_network(m: int, n: int, seed: int = None):
    """
    Generate a route network instance.
    
    :param m: Number of depots
    :param n: Number of trips
    :param seed: Random seed for reproducibility
    """
    # Set random seed
    if seed is not None:
        random.seed(seed)
    
    # Number of vehicles at each depot
    max_vehicles = round(3 + (n / (2 * m)))
    min_vehicles = round(3 + (n / (3 * m)))
    
    vehicles = {}
    for i in range(m):
        vehicles[i] = math.ceil(min_vehicles + random.random() * (max_vehicles - min_vehicles))
    
    # Number of locations
    nb_trip_locations = u_integer(math.ceil(n / 3), math.ceil(n / 2))
    
    # Initialize location coordinates
    l_coordinates = {}
    
    # Fix 4 depots at the four corners
    l_coordinates[0] = [0, 0]
    l_coordinates[1] = [60, 60]
    
    if m > 2:
        l_coordinates[2] = [0, 60]
    
    if m > 3:
        l_coordinates[3] = [60, 0]
    
    # Additional depots if more than 4
    if m > 4:
        for i in range(4, m):
            while True:
                x = random.randint(0, 59)
                y = random.randint(0, 59)
                
                # Check if coordinates are unique
                if not any(x == coord[0] and y == coord[1] for coord in l_coordinates.values()):
                    l_coordinates[i] = [x, y]
                    break
    
    # Generate additional trip locations
    location_number = m
    for _ in range(nb_trip_locations):
        while True:
            x = random.randint(0, 59)
            y = random.randint(0, 59)
            
            # Check if coordinates are unique
            if not any(x == coord[0] and y == coord[1] for coord in l_coordinates.values()):
                l_coordinates[location_number] = [x, y]
                location_number += 1
                break
    
    # Calculate distances between locations
    nb_locations = len(l_coordinates)
    distance_matrix = [[distance_calculate(l_coordinates[i][0], l_coordinates[i][1], 
                                            l_coordinates[j][0], l_coordinates[j][1]) 
                        for j in range(nb_locations)] 
                       for i in range(nb_locations)]
    
    # Generate trips
    trip_information = []
    for _ in range(n):
        prob = random.random()
        if prob < 0.6:
            # Round trip
            start_location = u_integer(m, nb_locations - 1)
            trip_time = u_integer(300, 1200)
            trip_information.append([
                start_location, 
                trip_time, 
                start_location, 
                u_integer(trip_time + 180, trip_time + 300)
            ])
        else:
            # Different start and end locations
            start_location = u_integer(m, nb_locations - 1)
            prob1 = random.random()
            
            # Start time distribution
            if prob1 < 0.15:
                start_time = u_integer(420, 480)
            elif prob1 < 0.85:
                start_time = u_integer(480, 1020)
            else:
                start_time = u_integer(1020, 1080)
            
            end_location = u_integer(m, nb_locations - 1)
            trip_length = distance_matrix[start_location][end_location]
            
            lower = math.ceil(start_time + trip_length + 5)
            higher = math.ceil(start_time + trip_length + 40)
            end_time = u_integer(lower, higher)
            
            trip_information.append([start_location, start_time, end_location, end_time])
    
    # Create output filename
    filename = f"RN-{m}-{n}-0{seed}.dat"
    
    # Write to file
    with open(filename, 'w') as f:
        # First line: number of depots, trips, total locations
        f.write(f"{m} {n} {nb_locations}\n")
        
        # Second line: number of vehicles per depot
        f.write(" ".join(str(vehicles[i]) for i in range(m)) + "\n")
        
        # Trips information
        for trip in trip_information:
            f.write(" ".join(map(str, trip)) + "\n")
        
        # Distance matrix
        for row in distance_matrix:
            f.write(" ".join(map(str, row)) + "\n")
    
    print(f"Generated Instance file: {filename}")

def main():
    # Example usage
    if len(os.sys.argv) > 3:
        m = int(os.sys.argv[1])
        n = int(os.sys.argv[2])
        nb_instances = int(os.sys.argv[3])
        
        for seed in range(1, nb_instances + 1):
            print(f"Generating {m} {n} {seed}")
            generate_route_network(m, n, seed)
    else:
        print("Usage: python script.py <num_depots> <num_trips> <num_instances>")

if __name__ == "__main__":
    main()