import torch
import geomstats.backend as gb
import geomstats.geometry.hypersphere as Hypersphere
import matplotlib.pyplot as plt
import numpy as np

# We create a 2D hypersphere, dim here is the manifold dimension ( so the sphere is embedded in R^3 )
sphere = Hypersphere.Hypersphere(dim=2)

# Now lets create data points on the sphere

x_0 = sphere.random_uniform() # noise ( start )  
x_1 = sphere.random_uniform() # target data ( end )

print("Start point (x_0):", x_0)
print("Target point (x_1):", x_1)

# calculate the velocity using log map
# This gives us the tangent vector at x_0 that points towards x_1
target_v = sphere.metric.log(point=x_1, base_point=x_0)


print("Tangent vector at x_0 pointing towards x_1 (target_v):", target_v)


# Simulate the flow (geodesic)
# we want to move from x_0 in the direction of target_v along the curve
# In flat space: x_t = (1-t)*x_0 + t*x_1
# On sphere : We use geodesic function
geodesic_func = sphere.metric.geodesic(initial_point=x_0, initial_tangent_vec=target_v)

t = 0.5 
x_t = geodesic_func(t)
print(f"Midpoint location (t=0.5): {x_t}")

# verify that x_t is indeed on the sphere
print(f"Is it on the manifold? Norm: {torch.norm(torch.tensor(x_t))}")

## Parallel transport of tangent vector

transported_v = sphere.metric.parallel_transport(
    tangent_vec = target_v,
    base_point = x_0,
    end_point = x_t
)


print(f"Transported tangent vector at t=0.5: {transported_v}")



## Now for the plot
times = gb.linspace(0, 1, 100)
path_points = geodesic_func(times)


fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')


## Plot the sphere Wireframe 
u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
x = np.cos(u)*np.sin(v)
y = np.sin(u)*np.sin(v)
z = np.cos(v)
ax.plot_wireframe(x, y, z, color="lightgray", alpha=0.5)


# Plot the geodesic path
ax.plot(path_points[:, 0], path_points[:, 1], path_points[:, 2], color='red', label='Geodesic Flow', linewidth=3)
ax.scatter(x_0[0], x_0[1], x_0[2], color='blue', s=100, label='Noise')
ax.scatter(x_1[0], x_1[1], x_1[2], color='green', s=100, label='Target Data')

ax.legend()
plt.show()