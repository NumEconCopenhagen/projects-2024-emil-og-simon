import numpy as np
import matplotlib.pyplot as plt

def find_point(X, y, condition):
    filtered_points = [point for point in X if condition(point, y)]
    if not filtered_points:
        return None
    return min(filtered_points, key=lambda point: np.sqrt((point[0]-y[0])**2 + (point[1]-y[1])**2))

def compute_ABCD(X, y):
    A = find_point(X, y, lambda point, y: point[0] > y[0] and point[1] > y[1])
    B = find_point(X, y, lambda point, y: point[0] > y[0] and point[1] < y[1])
    C = find_point(X, y, lambda point, y: point[0] < y[0] and point[1] < y[1])
    D = find_point(X, y, lambda point, y: point[0] < y[0] and point[1] > y[1])
    
    return A, B, C, D

def barycentric_coordinates(A,B,C,y):
    denom = (B[1] - C[1])*(A[0] - C[0]) + (C[0] - B[0])*(A[1] - C[1])
    r1 = ((B[1] - C[1])*(y[0] - C[0]) + (C[0] - B[0])*(y[1] - C[1])) / denom
    r2 = ((C[1] - A[1])*(y[0] - C[0]) + (A[0] - C[0])*(y[1] - C[1])) / denom
    r3 = 1 - r1 - r2
    return r1, r2, r3

def is_inside_triangle(r1, r2, r3):
    return 0 <= r1 <= 1 and 0 <= r2 <= 1 and 0 <= r3 <= 1

def interpolate_or_nan(f, X, y):
    # Step 1: Compute A, B, C, and D. If not possible return `NaN`.
    A, B, C, D = compute_ABCD(X, y)
    
    if A is None or B is None or C is None or D is None:
        return np.nan
    
    # Step 2: If y is inside triangle ABC, return r^{ABC}_1 f(A) + r^{ABC}_2 f(B) + r^{ABC}_3 f(C).
    r1_ABC, r2_ABC, r3_ABC = barycentric_coordinates(A, B, C, y)
    if is_inside_triangle(r1_ABC, r2_ABC, r3_ABC):
        return r1_ABC * f(A) + r2_ABC * f(B) + r3_ABC * f(C)
    
    # Step 3: If y is inside triangle CDA, we want to return r^{CDA}_1 f(C) + r^{CDA}_2 f(D) + r^{CDA}_3 f(A).
    r1_CDA, r2_CDA, r3_CDA = barycentric_coordinates(C, D, A, y)
    if is_inside_triangle(r1_CDA, r2_CDA, r3_CDA):
        return r1_CDA * f(C) + r2_CDA * f(D) + r3_CDA * f(A)
    
    # Step 4: Otherwise we return `NaN`.
    return np.nan

def plot_q1(X, y, A, B, C, D):
    plt.figure(figsize=(8, 8))
    plt.scatter(X[:, 0], X[:, 1], c='blue', label='Points in X')
    plt.scatter(*y, c='red', label='Point y', zorder=5)
    
    if A is not None and B is not None and C is not None:
        plt.scatter(*A, c='green', label='Point A')
        plt.scatter(*B, c='green', label='Point B')
        plt.scatter(*C, c='green', label='Point C')
        plt.plot([A[0], B[0], C[0], A[0]], [A[1], B[1], C[1], A[1]], c='green', linestyle='--', label='Triangle ABC')
    
    if C is not None and D is not None and A is not None:
        plt.scatter(*D, c='purple', label='Point D')
        plt.plot([C[0], D[0], A[0], C[0]], [C[1], D[1], A[1], C[1]], c='purple', linestyle='--', label='Triangle CDA')
    
    plt.legend()
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Points, y and Triangles ABC & CDA')
    plt.grid(True)
    plt.show()


def plot_q2(X, y):
    A, B, C, D = compute_ABCD(X, y)
    
    if A is None or B is None or C is None or D is None:
        print("Cannot compute ABCD or y is outside all triangles.")
        return
    
    r1_ABC, r2_ABC, r3_ABC = barycentric_coordinates(A, B, C, y)
    r1_CDA, r2_CDA, r3_CDA = barycentric_coordinates(C, D, A, y)

    plt.figure(figsize=(8, 8))
    plt.scatter(X[:, 0], X[:, 1], c='blue', label='Points in X')
    plt.scatter(*y, c='red', label='Point y', zorder=5)
    
    plt.scatter(*A, c='green', label='Point A')
    plt.scatter(*B, c='green', label='Point B')
    plt.scatter(*C, c='green', label='Point C')
    plt.scatter(*D, c='purple', label='Point D')
    
    if is_inside_triangle(r1_ABC, r2_ABC, r3_ABC):
        plt.plot([A[0], B[0], C[0], A[0]], [A[1], B[1], C[1], A[1]], c='green', linestyle='--', label='Triangle ABC')
        print(f"Barycentric coordinates of y with respect to triangle ABC:")
        print(f"r1_ABC = {r1_ABC:.5f}, r2_ABC = {r2_ABC:.5f}, r3_ABC = {r3_ABC:.5f}")
        print(f"Barycentric coordinates of y with respect to triangle CDA:")
        print(f"r1_CDA = {r1_CDA:.5f}, r2_CDA = {r2_CDA:.5f}, r3_CDA = {r3_CDA:.5f}")
        print ("Hence, y is in ABC")
    
    if is_inside_triangle(r1_CDA, r2_CDA, r3_CDA):
        plt.plot([C[0], D[0], A[0], C[0]], [C[1], D[1], A[1], C[1]], c='purple', linestyle='--', label='Triangle CDA')
        print(f"Barycentric coordinates of y with respect to triangle ABC:")
        print(f"r1_ABC = {r1_ABC:.5f}, r2_ABC = {r2_ABC:.5f}, r3_ABC = {r3_ABC:.5f}")
        print(f"Barycentric coordinates of y with respect to triangle CDA:")
        print(f"r1_CDA = {r1_CDA:.5f}, r2_CDA = {r2_CDA:.5f}, r3_CDA = {r3_CDA:.5f}")
        print ("Hence, y is in CDA")
    
    plt.legend()
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Points, y and Triangles ABC & CDA with Barycentric Coordinates of y')
    plt.grid(True)
    plt.show()

def plot_q4(X, Y, f):
    plt.figure(figsize=(12, 10))

    for i, (y_x, y_y) in enumerate(Y):
        A, B, C, D = compute_ABCD(X, (y_x, y_y))
        approx_value = interpolate_or_nan(f, X, (y_x, y_y))
        true_value = f((y_x, y_y))

        plt.subplot(2, 3, i + 1)
        plt.scatter(X[:, 0], X[:, 1], c='blue', label='Points in X')
        plt.scatter(y_x, y_y, c='red', label='Point y', zorder=5)
        
        if A is not None and B is not None and C is not None:
            plt.scatter(A[0], A[1], c='green', label='Point A')
            plt.scatter(B[0], B[1], c='green', label='Point B')
            plt.scatter(C[0], C[1], c='green', label='Point C')
            plt.plot([A[0], B[0], C[0], A[0]], [A[1], B[1], C[1], A[1]], c='green', linestyle='--', label='Triangle ABC')
        
        if C is not None and D is not None and A is not None:
            plt.scatter(D[0], D[1], c='purple', label='Point D')
            plt.plot([C[0], D[0], A[0], C[0]], [C[1], D[1], A[1], C[1]], c='purple', linestyle='--', label='Triangle CDA')
        
        plt.title(f'Point y = ({y_x}, {y_y})')
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.grid(True)
        plt.legend()

        if not np.isnan(approx_value):
            plt.text(0.5, 0.5, f'True f(y): {true_value:.4f}\nApprox. f(y): {approx_value:.4f}', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
        else:
            plt.text(0.5, 0.5, 'Point y outside all triangles', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
    
    plt.tight_layout()
    plt.show()