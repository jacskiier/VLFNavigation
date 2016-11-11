import numpy as np
from numpy import matlib


def LlhToEcef(Llh):
    """LlhToEcef  Convert WGS-84 coordinates to ECEF

       Ecef = LlhToEcef(Llh) converts a position(s) expressed in WGS-84
       coordiantes to Earth-Centered Earth-Fixed coordiantes.

       :type   Llh: 3-by-N array with rows [lat (rad); lon (rad); hae (m)]
       :param  Llh: Each column is a WGS-84 position. If Llh is an N-by-3 array and N ~=
          3, Llh will be internally convert to a 3-by-N array but will output
          Ecef of size N-by-3.


        :return: Ecef: 3-by-N array with rows [x (m); y (m); z (m)]. Each column is an
          Earth-Centered Earth-Fixed position. If Llh was an N-by-3 array and
          N ~= 3, Ecef will convert to size N-by-3.
        :rtype: an array of converted coordinates

       REFERENCE
          WGS-84 Reference System (DMA report TR 8350.2)

       AUTHOR
          John Raquet Python by Joe Curro
          Autonomy and Navigation Technology (ANT) Center
          Air Force Institute of Technology"""

    transpose = 0

    Llh = np.array(Llh)
    if len(Llh.shape) > 2:
        raise ValueError('Input must be only 2 dimensions')
    if len(Llh.shape) == 2:
        (nx, ny) = Llh.shape
    else:
        nx = Llh.shape[0]
        ny = 1
        Llh = np.expand_dims(Llh, axis=1)

    if (nx != 3) and (ny == 3):
        Llh = Llh.transpose()
        transpose = 1
    elif (nx != 3) and (ny != 3):
        raise ValueError('LlhToEcef:InputSize', 'Llh must be of size 3-by-N or N-by-3.')

    lat = Llh[0, :]
    lon = Llh[1, :]
    alt = Llh[2, :]

    totalCoords = lat.shape[0]

    # initial conditions
    a = 6378137
    e2 = 0.00669437999013

    rn = a * np.ones(totalCoords) / np.sqrt(1 - e2 * (np.sin(lat)) ** 2)
    R = (rn + alt) * np.cos(lat)

    Ecef = np.zeros(Llh.shape)
    Ecef[0, :] = R * np.cos(lon)
    Ecef[1, :] = R * np.sin(lon)
    Ecef[2, :] = (rn * (1 - e2) + alt) * np.sin(lat)

    if transpose:
        Ecef = Ecef.transpose()

    return Ecef


def EcefToLlh(Ecef, method="accurate"):
    """EcefToLlh  Convert ECEF coordiantes to WGS-84

   Llh = EcefToLlh(Ecef) converts a position(s) expressed in Earth-Centered
   Earth-Fixed coordiantes to WGS-84 coordiantes using the accurate method.

   Llh = EcefToLlh(Ecef, method) converts a position(s) expressed in
   Earth-Centered Earth-Fixed coordiantes to WGS-84 coordiantes using the
   specified method.

   PARAMETERS
      Ecef: 3-by-N array with rows [x (m); y (m); z (m)]. Each column is an
      Earth-Centered Earth-Fixed position. If Llh is an N-by-3 array
      and N ~= 3, the same output will be of size N-by-3.

      method: (Optional) Conversion method as a string. Options are 'fast' and
      'accurate'. If this parameter is not provided the default method is
      'accurate'. 'fast' returns a soution accuracy of 1 mm. 'accurate'
      returns a solution accuracy near machine precision. 'fast' runs
      approximatley twice as fast as 'accurate'.

   RETURN
      Llh: 3-by-N array with rows [lat (rad); lon (rad); hae (m)]. Each
      column is a WGS-84 position. If Ecef was an N-by-3 array and
      N ~= 3, Llh will convert to size N-by-3.

   REFERENCE
      WGS-84 Reference System (DMA report TR 8350.2)

   AUTHOR
      John Raquet
      Autonomy and Navigation Technology (ANT) Center
      Air Force Institute of Technology"""

    if method == 'accurate':
        methodNum = 1
    elif method == 'fast':
        methodNum = 2
    else:
        raise ValueError('Invalid method specified: %s', method)

    Ecef = np.matrix(Ecef)

    if methodNum == 1:
        # Accurate method

        transpose = 0
        if len(Ecef.shape) == 2:
            (nx, ny) = Ecef.shape
        else:
            nx = Ecef.shape[0]
            ny = 1
            Ecef = np.expand_dims(Ecef, axis=1)
        if (nx == 3) and (ny != 3):
            Ecef = Ecef.transpose()
            transpose = 1
        elif (nx != 3) and (ny != 3):
            raise ValueError('Ecef must be of size 3-by-N or N-by-3.')
        NP = Ecef.shape[0]

        # WGS-84 Constants
        a = 6378137  # Semi-major radius (m)
        b = 6356752.3142  # Semi-minor radius (m)
        g0 = 9.7803267714  # Equatorial gravity (m/s^2)
        k = 0.00193185138639  # Equatorial gravity constant (.)
        e2 = 0.00669437999013  # Eccentricity squared (.)

        # Calculate measured position in meridianal plane
        # Pm = [sqrt(Ecef(1)^2 + Ecef(2)^2);  % Horizontal
        #     Ecef(3)];                       % Vertical

        Pm = np.hstack((np.sqrt(np.power(Ecef[:, 0], 2) + np.power(Ecef[:, 1], 2)), Ecef[:, 2]))
        Pm = np.matrix(Pm)
        # Determine starting point for iteration
        if methodNum == 1:
            phi0 = np.arctan(Pm[:, 1] / Pm[:, 0])
            h0 = np.zeros((NP, 1))
        else:
            phi0 = np.zeros((NP, 1))
            h0 = np.zeros((NP, 1))

        dP = a * np.ones((NP, 2))  # Preload difference vector to start iteration

        cnt = 0

        # while cnt < 20
        while (np.max(dP) > .0001) & (cnt < 20) or (cnt < 3):  # 100 um error

            # Calculate N, dN, Jacobian from phi0 and h0

            slat = np.sin(phi0)
            clat = np.cos(phi0)
            s2lat = np.multiply(slat, slat)

            Nden = 1 - e2 * s2lat

            N = a / np.sqrt(Nden)

            # Calculate initial position in meridianal plane
            #     P0 = [(N+h0)*clat;          % Horizontal
            #         (N*(1-e2)+h0)*slat];  % Vertical
            P0 = np.concatenate((np.multiply((N + h0), clat), np.multiply((N * (1 - e2) + h0), slat)), axis=1)
            P0 = np.matrix(P0)
            # Calculate residual
            dP = Pm - P0  # Meters

            # Calculate inverse Jacobian (transformation from residual to lat and
            # alt)

            k1 = 1 - e2 * s2lat
            k2 = np.sqrt(k1)

            #     A = [slat*(e2*a*clat*clat/k1/k2-a/k2-h0) clat;
            #          clat*(a*(1-e2)/k2 + h0 + a*e2*(1-e2)*s2lat/k1/k2) slat];
            A11 = np.multiply(slat, (np.divide(np.multiply(e2 * a * clat, clat), np.divide(k1, k2)) - a / k2 - h0))
            A12 = clat
            A21 = np.multiply(clat, (a * (1 - e2) / k2 + h0 + np.divide(a * e2 * (1 - e2) * s2lat, np.divide(k1, k2))))
            A22 = slat

            Adet = np.multiply(A11, A22) - np.multiply(A21, A12)
            Ainv1 = np.divide(np.hstack((A22, -A12)), (Adet * np.ones((1, 2))))
            Ainv2 = np.divide(np.hstack((-A21, A11)), (Adet * np.ones((1, 2))))
            dHa = np.multiply(Ainv1[:, 0], dP[:, 0]) + np.multiply(Ainv1[:, 1], dP[:, 1])
            dHb = np.multiply(Ainv2[:, 0], dP[:, 0]) + np.multiply(Ainv2[:, 1], dP[:, 1])

            phi0 = phi0 + dHa
            h0 = h0 + dHb

            cnt += 1
        if cnt == 20:
            raise Warning('EcefToLlh() using the accurate method did not converge in 20 steps.')

        lam = np.arctan2(Ecef[:, 1], Ecef[:, 0])

        Pwgs = np.array([phi0, lam, h0])

        if not transpose:
            Pwgs = Pwgs.transpose()
        if Pwgs.shape[0] == 1:
            Pwgs = Pwgs.squeeze(0)
        Llh = Pwgs


    else:
        # Fast Method

        transpose = 0
        if len(Ecef.shape) == 2:
            (nx, ny) = Ecef.shape
        else:
            nx = Ecef.shape[0]
            ny = 1
            Ecef = np.expand_dims(Ecef, axis=1)
        if (nx != 3) and (ny == 3):
            Ecef = Ecef.transpose()
            transpose = 1
        elif (nx != 3) and (ny != 3):
            raise ValueError('Ecef must be of size 3-by-N or N-by-3.')
            # Initial conditions
        a = 6378137
        e2 = 0.00669437999013
        b = 6356752.314
        epsilon = 1E-15

        lat = np.asmatrix(np.zeros((1, Ecef.shape[1])))
        lon = np.asmatrix(np.zeros((1, Ecef.shape[1])))
        hae = np.asmatrix(np.zeros((1, Ecef.shape[1])))

        #  Perform the tranformation
        w = np.sqrt(np.power(Ecef[0, :], 2) + np.power(Ecef[1, :], 2))

        i1Logical = abs(w) > epsilon
        i2Logical = abs(w) <= epsilon

        i1 = np.asarray(i1Logical).flatten()
        i2 = np.asarray(i2Logical).flatten()

        if np.any(i1Logical):
            l = e2 / 2
            l2 = l ** 2
            m = np.power(np.divide(w[i1Logical], a), 2)
            tmp1 = Ecef[2, i1] * (1 - e2) / b
            n = np.power(tmp1, 2)
            i = -0.5 * (2 * l2 + m + n)
            k = l2 * (l2 - m - n)
            tmp2 = m + n - 4 * l2
            mnl2 = np.multiply(m, n * l2)
            q = (np.power(tmp2, 3)) / 216 + mnl2
            D = np.sqrt(np.multiply((2 * q - mnl2), mnl2))
            beta = i / 3 - (np.power((q + D), (1 / 3.0))) - (np.power((q - D), (1 / 3.0)))
            tmp3 = np.sqrt(np.sqrt(np.power(beta, 2) - k) - 0.5 * (beta + i))
            t = tmp3 - np.multiply(np.sign(m - n), np.sqrt(0.5 * (beta - i)))
            w1 = np.divide(w[i1Logical], (t + l))
            z1 = np.multiply(Ecef[2, i1], (1 - e2) / (t - l))

            lat[:, i1] = np.arctan2(z1, (w1 * (1 - e2)))
            lon[:, i1] = 2 * np.arctan2((w[i1Logical] - Ecef[0, i1]), Ecef[1, i1])
            tmp4 = w[i1Logical] - w1
            tmp5 = Ecef[2, i1] - z1
            hae[:, i1] = np.multiply(np.sign(t - 1 + l), np.sqrt(np.power(tmp4, 2) + np.power(tmp5, 2)))

        if np.any(i2Logical):
            hae[:, i2] = np.sign(Ecef[2, i2]) * (Ecef[2, i2] - np.sign(Ecef[2, i2]) * b)
            lat[:, i2] = np.sign(Ecef[2, i2]) * np.pi / 2.0
            lon[:, i2] = 0

        iLogical = lon > np.pi
        i = np.asarray(iLogical).flatten()
        lon[:, i] = lon[:, i] - 2 * np.pi

        Llh = np.vstack((lat, lon, hae))

        if (transpose):
            Llh = Llh.transpose()

    if nx == 1:
        Llh = np.squeeze(Llh)

    return Llh


def LlhToCen(Llh):
    """QuatToRpy  Calculate a NED navigation frame to ECEF earth frame DCM from
           a WGS-84 position

   Cen = LlhToCen(Llh) converts from an WGS-84 (height above ellipsoid)
   position to a [north, east, down] navigation frame to [Earth-Centered,
   Earth-Fixed] Earth frame direction cosine matrix. Llh is a 3-by-N array and
   Cen is a 3-by-3-by-N array.

   PARAMETERS
      Llh: 3-by-N array where each N is a vector of [lat; lon; height above
      elipsoid] expressed in radians and meters. If Llh is an N-by-3 array
      and N ~= 3, Llh will be internally convert to a 3-by-N array and return
      the same output.

   RETURN
      Cen: 3-by-3-by-N array where each N is a [north, east, down] navigation
      frame to [Earth-Centered, Earth-Fixed] Earth frame direction cosine
      matrix. If N is 1, then Cen is a single 3-by-3 matrix.

   REFERENCE
      WGS-84 Reference System (DMA report TR 8350.2)

   AUTHOR
      Mark Smearcheck
      Autonomy and Navigation Technology (ANT) Center
      Air Force Institute of Technology

   SEE ALSO
      EcefToCen"""

    # Transpose Llh if provided as a series of rows instead of a series of
    # columns.
    Llh = np.asmatrix(Llh)

    nx, ny = Llh.shape

    if (nx != 3) and (ny == 3):
        Llh = Llh.transpose()
    elif (nx != 3) and (ny != 3):
        raise ValueError('LlhToCen:InputSize', 'Llh must be of size 3-by-N or N-by-3.')

    clat = np.cos(Llh[0, :])
    slat = np.sin(Llh[0, :])
    clon = np.cos(Llh[1, :])
    slon = np.sin(Llh[1, :])

    # Cen = np.zeros((3, 3, Llh.shape[1]))
    Cne = np.zeros((3, 3, Llh.shape[1]))
    Cne[0, 0, :] = np.multiply(slat, -clon)
    Cne[0, 1, :] = np.multiply(slat, -slon)
    Cne[0, 2, :] = clat * 1
    Cne[1, 0, :] = 1 * -slon
    Cne[1, 1, :] = 1 * clon
    Cne[1, 2, :] = 0
    Cne[2, 0, :] = np.multiply(clat, -clon)
    Cne[2, 1, :] = np.multiply(clat, -slon)
    Cne[2, 2, :] = -slat * 1

    # Cen = permute(Cne, [2, 1, 3]); # Transpose each 3-by-3 in the 3-by-3-by-N array.
    Cen = np.transpose(Cne, (1, 0, 2))

    if Llh.shape[1] == 1:
        Cen = Cen.squeeze(2)
    return Cen


def EcefToCen(Ecef):
    """EcefToCen  Calculate a NED navigation frame to ECEF earth frame DCM from
           an ECEF position

   Cen = EcefToCen(Ecef) converts from an Earth-Centered, Earth-Fixed position
   to a [north, east, down] navigation frame to [Earth-Centered, Earth-Fixed]
   Earth frame direction cosine matrix. Ecef is a 3-by-N array and Cen is a
   3-by-3-by-N array.

   PARAMETERS
       Ecef: 3-by-N array where each N is a vector of [ECEF-x; ECEF-y; ECEF-z]
       expressed in meters. If Ecef is an N-by-3 array and N ~= 3, Ecef will
       be internally convert to a 3-by-N array and return the same output.

   RETURN
       Cen: 3-by-3-by-N array where each N is a [north, east, down] navigation
       frame to [Earth-Centered, Earth-Fixed] Earth frame direction cosine
       matrix. If N is 1, then Cen is a single 3-by-3 matrix.

   REFERENCE
      WGS-84 Reference System (DMA report TR 8350.2)

   AUTHOR
      Mark Smearcheck
      Autonomy and Navigation Technology (ANT) Center
      Air Force Institute of Technology"""

    Ecef = np.asmatrix(Ecef)
    (nx, ny) = Ecef.shape
    if (nx != 3) and (ny == 3):
        Ecef = Ecef.transpose()
    elif (nx != 3) and (ny != 3):
        raise ValueError('EcefToCen:InputSize', 'Ecef must be of size 3-by-N or N-by-3.')

    Llh = EcefToLlh(Ecef)
    Cen = LlhToCen(Llh)
    return Cen


def EcefToLocalLevel(Ecef0, Ecef):
    """EcefToLocalLevel  Convert ECEF coordiantes to NED local level

   LocalLevel = EcefToLocalLevel(Ecef0, Ecef) converts a position(s) expressed
   in Earth-Centered Earth-Fixed coordiantes to [north, east, down] local
   level coordiantes.

   PARAMETERS
      Ecef0: 3-by-1 array of the reference position expressed in
      Earth-Centered Earth-Fixed coordinates with rows [x0 (m); y0 (m);
      z0 (m)]. If Ecef0 is an 1-by-3 array, Ecef0 will be internally convert
      to a 3-by-1 array.

      Ecef: 3-by-N array with rows [x (m); y (m); z (m)]. Each column is an
      Earth-Centered Earth-Fixed position. If Ecef is an N-by-3 array and N ~=
      3, Ecef will be internally convert to a 3-by-N array but will output
      LocalLevel of size N-by-3.

   RETURN
      LocalLevel: 3-by-N array with rows [north (m); east (m); down (m)]. Each
      column is a [north, east, down] local level position. If Ecef was an
      N-by-3 array and N ~= 3, LocalLevel will convert to size N-by-3.

   REFERENCE

   AUTHOR
      Mark Smearcheck
      Autonomy and Navigation Technology (ANT) Center
      Air Force Institute of Technology
"""

    transpose = 0
    Ecef = np.asmatrix(Ecef)
    Ecef0 = np.asmatrix(Ecef0)
    if len(Ecef.shape) > 2:
        raise ValueError('Input must be only 2 dimensions')
    (nx, ny) = Ecef.shape
    if (nx != 3) and (ny == 3):
        Ecef = Ecef.transpose()
        transpose = 1
    elif (nx != 3) and (ny != 3):
        raise ValueError('LlhToEcef:InputSize', 'Llh must be of size 3-by-N or N-by-3.')

    (nx, ny) = Ecef0.shape
    if (nx != 3) and (ny == 3):
        Ecef0 = Ecef0.transpose()

    N = Ecef.shape[1]

    Ecef0 = matlib.repmat(Ecef0, 1, N)
    Cen = EcefToCen(Ecef0)
    if len(Cen.shape) < 3:
        Cen = np.expand_dims(Cen, axis=2)
    dEcef = Ecef - Ecef0
    LocalLevel = np.asmatrix(np.zeros((3, N)))
    for i in range(N):
        LocalLevel[:, i] = Cen[:, :, i].transpose() * dEcef[:, i]  # omg an actual matrix multiply

    # If Ecef was input as a series of rows output LocalLevel as a series of
    # rows.
    if transpose:
        LocalLevel = LocalLevel.transpose()

    return LocalLevel


if __name__ == '__main__':
    llh = np.array([39.7825209064 * np.pi / 180.0, -84.0831101536 * np.pi / 180.0, 294.168640137])
    ecef = np.array([505988.185649, -4882270.332984, 4059646.518246])

    llh2 = np.array([39.7826120906 * np.pi / 180.0, -84.0830598574 * np.pi / 180.0, 294.168640137])
    ecef2 = np.array([505991.803621, -4882263.444778, 4059654.298851])

    # Test LlhToEcef
    ecefEst = LlhToEcef(llh)
    ecefEst = np.transpose(ecefEst)
    assert np.linalg.norm(ecefEst - ecef) < 1, "LlhToEcef failed to convert correctly {0} to the True:{1}".format(ecefEst, ecef)
    # test output array
    ecefArray = LlhToEcef([llh, llh2])
    ecefEst = np.transpose(ecefEst)
    assert ecefArray.shape[1] > 1, "LlhToEcef failed to output an array "

    # Test EcefToLlh
    llhEst = EcefToLlh(ecef)
    llhEst = np.transpose(llhEst)
    assert np.linalg.norm(llhEst - llh) < 1, "EcefToLlh failed to convert correctly {0} to the True:{1}".format(llhEst, llh)

    # llhEst = EcefToLlh(gpsCoords, method = "fast")
    # llhEst = np.transpose(llhEst)
    # assert np.linalg.norm(llhEst - llh) < 1, "EcefToLlh failed to convert correctly {0} to the True:{1}".format(llhEst, llh)

    # test output array
    llhEstArray = EcefToLlh([ecef, ecef2])
    assert llhEstArray.shape[1] > 1, "EcefToLlh failed to output an array"
    assert np.linalg.norm(llhEstArray[0, :] - llh) < 1 and np.linalg.norm(
        llhEstArray[1, :] - llh2) < 1, "EcefToLlh failed to convert correctly {0} to the True:{1}".format(llhEst, llh)

    # test output array
    llhEstArray = EcefToLlh([ecef, ecef2], "fast")
    assert llhEstArray.shape[1] > 1, "EcefToLlh failed to output an array"
    assert np.linalg.norm(llhEstArray[0, :] - llh) < 1 and np.linalg.norm(
        llhEstArray[1, :] - llh2) < 1, "EcefToLlh failed to convert correctly {0} to the True:{1} or {2} to {3}".format(llhEstArray[0, :],
                                                                                                                        llh, llhEstArray[1, :], llh2)

    # test LlhToCen
    dcm = np.asarray([[-0.0660, 0.9947, -0.0792], [0.6365, 0.1031, 0.7644], [0.7685, 0, -0.6399]])
    dcmEst = LlhToCen(llh)
    assert np.linalg.norm(dcm - dcmEst) < 1, "the dcm was found wrong"

    # test an array of llh
    dcm2 = np.asarray([[-0.0660, 0.9947, -0.0792], [0.6365, 0.1031, 0.7644], [0.7685, 0, -0.6399]])
    dcmArrayEst = LlhToCen([llh, llh2])
    assert np.linalg.norm(dcm - dcmArrayEst[:, :, 0]) < 1 and np.linalg.norm(dcm2 - dcmArrayEst[:, :, 1]) < 1, "the dcm was found wrong"

    # test EcefToCen
    dcmEst = EcefToCen(ecef)
    assert np.linalg.norm(dcm - dcmEst) < 1, "the dcm was found wrong"

    # test an array
    dcmArrayEst = EcefToCen([ecef, ecef2])
    assert np.linalg.norm(dcm - dcmArrayEst[:, :, 0]) < 1 and np.linalg.norm(dcm2 - dcmArrayEst[:, :, 1]) < 1, "the dcm was found wrong"

    localLevel = np.array([10.1247, 4.3088, 0.0000])
    localLevelEst = EcefToLocalLevel(ecef, ecef2)
    assert np.linalg.norm(localLevel - localLevelEst) < 1, "local level was wrong {0} to {1}".format(localLevelEst, localLevel)
