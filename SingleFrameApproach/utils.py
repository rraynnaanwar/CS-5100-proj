import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Arc, Circle
import math


def get_ball_coords(transformed_bbox_info): 
    for item in transformed_bbox_info:
        if 'Ball' in item:
            return item['Ball']
    return None

def calculate_distance_to_goal(x, y):
    # Goal center is the midpoint of the two goalposts
    goal_center_x = 45  # Midpoint of the goalposts at x = 45
    goal_center_y = 0   # Goal line is at y = 0 (for simplicity)

    # Use the Euclidean distance formula to calculate distance from ball to goal center
    distance = math.sqrt((x - goal_center_x)**2 + (y - goal_center_y)**2)
    return distance

def plot_shot(x, y, shot_angle=None):
    # Pitch dimensions (as per your specification)
    pitch_width = 90
    pitch_length = 120
    half_pitch_length = pitch_length / 2  # 60m
    
    # Goal dimensions (using your exact coordinates)
    g0 = np.array([41.34, 0])  # Left goalpost (x=41.34, y=0)
    g1 = np.array([48.66, 0])  # Right goalpost (x=48.66, y=0)
    goal_width = g1[0] - g0[0]  # 7.32m 
    goal_height = 2.44
    
    distance = calculate_distance_to_goal(x,y)

    # Shot position
    p = np.array([x, y])
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Draw pitch (green rectangle) - only half
    ax.add_patch(Rectangle((0, 0), pitch_width, half_pitch_length, 
                         fill=True, color='#3a5a0b', zorder=1))
    
    # ------ Pitch Markings ------
    # Penalty area (16.5m deep, 40.32m wide including goal)
    penalty_area_length = 16.5
    penalty_area_width = 40.32
    ax.add_patch(Rectangle((g0[0]-16.5, 0), penalty_area_width, penalty_area_length, 
                 fill=False, color='white', linewidth=2, zorder=2))
    
    # 6-yard box (5.5m deep, 18.32m wide including goal)
    six_yard_length = 5.5
    six_yard_width = 18.32
    ax.add_patch(Rectangle((g0[0]-5.5, 0), six_yard_width, six_yard_length, 
                         fill=False, color='white', linewidth=2, zorder=2))
    
    # Goal
    ax.plot([g0[0], g0[0]], [0, goal_height], color='red', linewidth=3, zorder=3)
    ax.plot([g1[0], g1[0]], [0, goal_height], color='red', linewidth=3, zorder=3)
    ax.plot([g0[0], g1[0]], [goal_height, goal_height], color='red', linewidth=3, zorder=3)
    
    # Penalty spot (11m from goal line)
    penalty_spot = np.array([pitch_width/2, 11])
    ax.add_patch(Circle((penalty_spot[0], penalty_spot[1]), 0.5, color='white', zorder=2))
    
    # Center circle (only showing half)
    center_circle_radius = 9.15
    ax.add_patch(Arc((pitch_width/2, half_pitch_length), 
                   width=2*center_circle_radius, height=2*center_circle_radius,
                   angle=0, theta1=180, theta2=360, color='white', linewidth=2))
    
    # ------ Shot Visualization ------
    # Mark shot position
    ax.scatter(p[0], p[1], color='blue', s=100, label='Shot Position', zorder=5)
    
    # Draw lines to goalposts
    ax.plot([p[0], g0[0]], [p[1], g0[1]], 'b--', alpha=0.5, zorder=4)
    ax.plot([p[0], g1[0]], [p[1], g1[1]], 'b--', alpha=0.5, zorder=4)
    
    info_text = f"Distance: {distance:.1f}m"
    # Display viewing angle if provided
    if shot_angle is not None:
        info_text += f"\nAngle:{shot_angle:.1f}°"
    ax.text(x + 2, y, info_text, 
           fontsize=12, color='white', weight='bold', zorder=5,
           bbox=dict(facecolor='navy', alpha=0.7, edgecolor='white'))
    
    
    # ------ Plot Formatting ------
    ax.set_xlim(-5, pitch_width + 5)
    ax.set_ylim(-2, half_pitch_length + 5)  # Showing only half pitch
    ax.set_aspect('equal')
    ax.set_facecolor('#2e6b34')  # Darker green for out-of-pitch area
    ax.set_title("Football Pitch - Shot Visualization", fontsize=14, pad=20)
    ax.set_xlabel("Width (meters)", fontsize=10)
    ax.set_ylabel("Length (meters)", fontsize=10)
    ax.grid(False)
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.show()



