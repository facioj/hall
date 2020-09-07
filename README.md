# hall
Implementation of 
1) functions that compute summed-over-bands Berry curvature or Berry curvature dipole at given k-point (hall_k class)
2) functions that based on such methods computes a 3D integral using a slicing of the Brillouin zone method (slicemesh class)

The spirit is that hall_k could be easily interfaced with other methods that perform the 3D integration, such as the adaptive mesh method, and that slicemesh could be easily adapted to integrate different functions.
