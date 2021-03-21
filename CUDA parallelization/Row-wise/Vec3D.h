
#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#else
#define CUDA_HOSTDEV
#endif

#ifndef CUDAADV_VEC3D_H
#define CUDAADV_VEC3D_H
#define V3DFMT "%d,%d,%d"
#define V3DFMTSZ "%dx%dx%d"
#define V3DLST(v) (v).x,(v).y,(v).z
#include <algorithm>

template <class T>
class Vec3D {
public:
    union {
        struct {T x, y, z;};
        T v[3];
    };

    CUDA_HOSTDEV Vec3D<T> (T xv, T yv, T zv){
        x = xv;
        y = yv;
        z = zv;
    }

    CUDA_HOSTDEV Vec3D<T> (T v){
        x = y = z = v;
    }

    CUDA_HOSTDEV Vec3D<T> () {
        x = y = z = 0;
    }

    Vec3D<T> min(Vec3D<T> v){
        return Vec3D<T> (std::min(x, v.x), std::min(y, v.y), std::min(z, v.z));
    }

    Vec3D<T> max(Vec3D<T> v){
        return Vec3D<T> (std::max(x, v.x), std::max(y, v.y), std::max(z, v.z));
    }

    CUDA_HOSTDEV T prod(){
        return x * y * z;
    }

    CUDA_HOSTDEV T sum(){
        return x + y + z;
    }

    CUDA_HOSTDEV Vec3D<T> operator + (const Vec3D<T>& v) const {
        return Vec3D<T>  (x + v.x, y + v.y, z + v.z);
    }
    CUDA_HOSTDEV Vec3D<T> operator + (int v) const {
        return Vec3D<T>  (x + v, y + v, z + v);
    }
    CUDA_HOSTDEV Vec3D<T> operator - (const Vec3D<T>& v) const {
        return Vec3D<T>  (x - v.x, y - v.y, z - v.z);
    }
    CUDA_HOSTDEV Vec3D<T> operator - (int v) const {
        return Vec3D<T>  (x - v, y - v, z - v);
    }
    CUDA_HOSTDEV Vec3D<T> operator * (const Vec3D<T>& v) const {
        return Vec3D<T>  (x * v.x, y * v.y, z * v.z);
    }
    CUDA_HOSTDEV Vec3D<T> operator * (int v) const {
        return Vec3D<T>  (x * v, y * v, z * v);
    }
    CUDA_HOSTDEV Vec3D<T> operator / (const Vec3D<T>& v) const {
        return Vec3D<T>  (x / v.x, y / v.y, z / v.z);
    }
    friend Vec3D<T> operator / (int a, const Vec3D<T>& v) {
        return Vec3D<T>  (a / v.x, a / v.y, a / v.z);
    }
    CUDA_HOSTDEV Vec3D<T> operator % (const Vec3D<T>& v) const {
        return Vec3D<T>  (x % v.x, y % v.y, z % v.z);
    }
    friend bool operator == (const Vec3D<T>& u, const Vec3D<T>& v) {
        return (u.x == v.x && u.y == v.y && u.z == v.z);
    }
    friend bool operator <= (const Vec3D<T>& u, const Vec3D<T>& v) {
        return (u.x <= v.x && u.y <= v.y && u.z <= v.z);
    }

};

#endif //CUDAADV_VEC3D_H