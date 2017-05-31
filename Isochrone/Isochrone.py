#!/usr/bin/env python
"""
Isochrone Modules that calculates the Isochrone of a 
two dimensional steady flow computed using MODFLOW2005

Requires: numpy, scipy, matplotlib and flopy
Optional: shapefile

Authors: A. Feo, A. Zanini, E. Petrella, F. Celico, (2017)
Parma University

Based on:
 "Scripting MODFLOW Model Development Using Python and FloPy. Groundwater"
     Bakker, M., Post, V., Langevin, C. D., Hughes, J. D., White, J. T., 
     Starn, J. J. and Fienen, M. N. 
  (2016), doi:10.1111/gwat.12413

"""

__version__ = '1.0.0'

import sys
import os
import numpy as np
import scipy
import scipy.interpolate
import matplotlib
import matplotlib.pylab as plt

import flopy.modflow          as flopy_mf
import flopy.utils            as flopy_ut
import flopy.utils.binaryfile as flopy_bf

try:
    import shapefile
    SHAPEFILES = True
except:
    SHAPEFILES = False


#
#  Arrays are stored using the following convention
#
#  data[ k , j , i ]
#        |   |   |
#        |   |    ---> (x index, column)  
#        |   --------> (y index, row)
#        ------------> (z index, layer)
#
#
#  One should keep in mind that the convention has
#  the effect that increasing the j-index correspond
#  to a decrease of the y-coordinate 
#
#  shape of the grid = (ml.dis.nlay,ml.dis.nrow,ml.dis.ncol)
#
#  index start from 0 not 1 (different the fortran mf2005 convention)
# 

class SteadyFlow :
    """Class to analyze the results and the isochrone of a MODFLOW2005 
    simulation in the case of a steady state flow condition """

    def __init__ (self, modelname, time=1.0, LAYER=0,
                  UPPERLEFTx=0.0,UPPERLEFTy=0.0, Alpha=0.0,
                  head_file_name='',budget_file_name='',
                  precision = 'single') :
        """ Initialize the steady flow case from a MODFLOW simulation output
     
        Keyword arguments:

        time -- initial time of the simulation (default 1.0) 
        LAYER -- layer to be considered for the 2d flow (default 0)

        UPPERLEFTx -- x-position in the upper left corner (default 0.0)
        UPPERLEFTy -- y-position in the upper left corner (default 0.0)
        Alpha -- the angle in degree for the roto-translation (default 0.0)

        head_file_name   --  name of the head file (default modelname+.hds)
        budget_file_name --  name of the budget file (default modelname+.cbc)

        precision = precison of the binary files (default 'single') 
        """

        self.precision = precision
        # Convert the angle from degree to radiant 
        alpha = Alpha/180.0*np.pi  
        self.LAYER = LAYER
        self.Rot = np.array([[np.cos(alpha), -np.sin(alpha)],
                            [np.sin(alpha), np.cos(alpha)]])
        self.Origin = np.array([UPPERLEFTx,UPPERLEFTy])

        # Get the name without extension
        self.name, file_extension = os.path.splitext(modelname)
        if (len(file_extension) < 2) :
            ml = flopy_mf.Modflow.load(self.name + '.nam')
        else:
            ml = flopy_mf.Modflow.load(modelname)

        # Get the properties of the MODFLOW simulation
        y, x, z = ml.dis.get_node_coordinates()

        # In FloPy the origin of x,y is in the bottom left 
        # instead of the upper left of the grid (as in MODFLOW)
        NCOL = ml.dis.ncol     # convention x-direction
        NROW = ml.dis.nrow     # convention y-direction
        NLAY = ml.dis.nlay     # convention z-direction
        FLOWshape = (NLAY, NROW, NCOL)

        # (NCOL)  spacing in the x-direction
        # (NROW)  spacing in the y-direction
        # (NLAY,NROW,NCOL)
        deltaROW = ml.dis.delr.array 
        deltaCOL = ml.dis.delc.array
        THICKNESS = ml.dis.thickness.array[LAYER, :, :] 

        # TOP       = ml.dis.top.array       # (NROW,NCOL)
        # BOTTOM    = ml.dis.bottom.array    # (NLAY,NROW,NCOL)
        xP = 0.0 + np.cumsum(deltaROW)
        yP = 0.0 - np.cumsum(deltaCOL)
        xM = xP - deltaROW[0]
        yM = yP + deltaCOL[0]
        self.bbox = [0.0, xP[-1], yP[-1], 0.0]
        xC = 0.5 * (xM+xP)
        yC = 0.5 * (yM+yP)

        # X ,Y  (matrix) of center-cell coordinates
        # Xr, Yr (matrix) of right-face  coordinates
        # Xf, Yf (matrix) of front-face  coordinates
        self.X, self.Y = np.meshgrid(xC, yC)  
        self.Xr, self.Yr = np.meshgrid(xP, yC)
        self.Xf, self.Yf = np.meshgrid(xC, yP)

        # x, y  (vector) of center-cell coordinates
        # xr, yr (vector) of right-face  coordinates
        # xf, yf (vector) of front-face  coordinates
        self.x, self.y = xC, yC      
        self.xr, self.yr = xP, yC    
        self.xf, self.yf = xC, yP    
        self.deltaCOL = deltaCOL  
        self.deltaROW = deltaROW  
        self.THICKNESS = THICKNESS 
        self.FLOWshape = FLOWshape
        self.ml = ml

        # Read binary data for head and budget
        if len(head_file_name) > 0: 
           self.HEAD_FILE_NAME = head_file_name
        else:
           self.HEAD_FILE_NAME = self.name + '.hds' 
        if len(budget_file_name) > 0: 
            self.BUDGET_FILE_NAME = budget_file_name
        else: 
            self.BUDGET_FILE_NAME = self.name + '.cbc'

        # Read head data
        hds = flopy_bf.HeadFile(self.HEAD_FILE_NAME,precision = precision)
        self.head = hds.get_data(totim=time)[LAYER, :, :]
        hds.close()

        # Read budget data  (Flows on the faces and on wells)
        cbb = flopy_bf.CellBudgetFile(self.BUDGET_FILE_NAME,
                                      precision = precision)
        self.FLOWx = cbb.get_data(totim=time,
                                  text='FLOW RIGHT FACE ')[0][LAYER,:,:]
        self.FLOWy = cbb.get_data(totim=time,text='FLOW FRONT FACE ')[0][LAYER,:,:]
        wells = cbb.get_data(totim=time, text='           WELLS')[0]
        WELLs = {'N':0, 'Q':[], 'idx':[]}

        # WELLs['N']   --> Number of well present
        # WELLs['idx'] --> 3d index of well cell (MODFLOW)
        # WELLs['Q']   --> Q of each WELL
        for idx in range(len(wells)) :
            idx3d = np.unravel_index(wells[idx][0]-1, FLOWshape)
            if idx3d[0] == LAYER :
                WELLs['N'] = WELLs['N'] + 1
                WELLs['Q'].append(wells[idx][1])
                WELLs['idx'].append(idx3d)
        self.WELLs = WELLs
        cbb.close()

    # Rototranslate a point
    def RotoTranslate(Point) :
        """ Utility that perform a rototranslation of a point using the 
        origin and angle associated to the model
        """
        Xnew = self.Origin[0]+self.M[0, 0]*Point[0]+self.M[0, 1]*Point[1]
        Ynew = self.Origin[1]+self.M[1, 0]*Point[0]+self.M[1, 1]*Point[1]

    # Compute velocity fields (interpolator)
    def ComputeVelocityFieldInterpolator(self, porosity=0.3,
                                         interp = 'RectBivariateSpline') :
        """Compute the velociy and porosity interpolator of the 2 dimensional flow
        of the SteadyFlow model

        Keyword arguments:

        porosity -- porosity of the cell (default constant porosity 0.3)
                    This parameter can also be:
                    1) A file name that contains the porosity of each cell
                    2) An array of values containing the porosities. 
        interp -- interpolator used to integrate flows (default 'RectBivariateSpline')
                  there is the alternative option of 'linear' interpolator 
 
        """
        NROW, NCOL = self.FLOWx.shape

        print(porosity)
        if (type(porosity) == str) :
            if (os.path.exists(porosity) == True) :
                datas = np.loadtxt(porosity)
                len_datas = datas.size
                print("The number of porosity data read is: " + str(len_datas))
                if len_datas == np.prod(self.FLOWshape) : 
                    # The size is the same of (NLAY,NROW,NCOL)
                    datas = datas.flatten().reshape(self.FLOWshape,order='C')
                    Porosity = np.ones(self.FLOWx.shape)
                    Porosity[:, :] = datas[self.LAYER, :, :] 
                elif len_datas == np.prod(self.FLOWx.shape) : 
                    # The size is the same of (NROW,NCOL)
                    datas = datas.flatten().reshape(self.FLOWx.shape, order='C')
                    Porosity = np.ones(self.FLOWx.shape)
                    Porosity[:, :] = datas[:, :] 
                elif len_datas == 1 : 
                    # Just one value
                    Porosity = datas[0] * np.ones(self.FLOWx.shape)   
                else:
                    print("Shape of porosity file does not match the grid assuming porosity = 0.3")
                    Porosity = 0.3 * np.ones(self.FLOWx.shape)   
            else :
                print("Porosity file ", porosity, "does not exist ! assuming porosity = 0.3")
                Porosity = 0.3 * np.ones(self.FLOWx.shape)
        elif (type(porosity) == np.ndarray) :
            Porosity = np.ones(self.FLOWx.shape)
            Porosity[:, :] = porosity[:, :] 
        else :
            Porosity = porosity * np.ones(self.FLOWx.shape)

        Qx = np.zeros(self.FLOWx.shape)
        Qy = np.zeros(self.FLOWx.shape)
        for j in range(NROW) : 
            for i in range(NCOL) : 
                # Qx[j,i] = (1.0/(porosity*self.THICKNESS[j,i]*self.deltaCOL[j])) * self.FLOWx[j,i]
                # Qy[j,i] = (1.0/(porosity*self.THICKNESS[j,i]*self.deltaROW[i])) * self.FLOWy[j,i]
                Qx[j,i] = (1.0/(self.THICKNESS[j, i] * self.deltaCOL[j])) * self.FLOWx[j, i]
                Qy[j,i] = (1.0/(self.THICKNESS[j, i] * self.deltaROW[i])) * self.FLOWy[j, i]
        self.maxQ = np.max(np.abs(Qx)) + np.max(np.abs(Qy))

        # Now we use the interpolator
        if interp == 'RectBivariateSpline' :
            INTERP_Vx =  Qx[::-1, :]
            INTERP_Vy = -Qy[::-1, :]
            Fvelx = scipy.interpolate.RectBivariateSpline(self.xr, self.yr[::-1], INTERP_Vx.T, bbox=self.bbox)
            Fvely = scipy.interpolate.RectBivariateSpline(self.xf, self.yf[::-1], INTERP_Vy.T, bbox=self.bbox)
            Fpor  = scipy.interpolate.RectBivariateSpline(self.x, self.y[::-1], Porosity.T, bbox=self.bbox)
            self.Vx = ((Fvelx(self.x, self.y[::-1]) ).T)[::-1, :]
            self.Vy = ((Fvely(self.x, self.y[::-1]) ).T)[::-1, :]
            self.por= ((Fpor(self.x, self.y[::-1]) ).T)[::-1, :]
            self.maxV = np.sqrt(np.max(self.Vx**2 + self.Vy**2))
            self.f_vel = lambda P : np.array([Fvelx(P[0], P[1])[0, 0], Fvely(P[0], P[1])[0, 0]])
            self.f_por = lambda P : Fpor(P[0], P[1])[0, 0]
        elif interp =='linear' :
            INTERP_Vx =  Qx[::-1, :]
            INTERP_Vy = -Qy[::-1, :]
            Fvelx = scipy.interpolate.RegularGridInterpolator((self.xr, self.yr[::-1]), INTERP_Vx.T,
                                                              method='linear', bounds_error=False, fill_value = 0.0)
            Fvely = scipy.interpolate.RegularGridInterpolator((self.xf, self.yf[::-1]), INTERP_Vy.T,
                                                              method='linear', bounds_error=False, fill_value = 0.0)
            Fpor  = scipy.interpolate.RegularGridInterpolator((self.x, self.y[::-1]), Porosity.T,
                                                              method='nearest', bounds_error=False, fill_value = 0.0)
            self.Vx = ((Fvelx((self.X, self.Y[::-1])) ))[::-1, :]
            self.Vy = ((Fvely((self.X, self.Y[::-1])) ))[::-1, :]
            self.por= ((Fpor((self.X, self.Y[::-1])) ))[::-1, :]
            self.f_vel = lambda P : np.array([Fvelx(P), Fvely(P)])[:, 0]
            self.f_por = lambda P : Fpor([P[0], P[1]])[0]

        # self.f_ode    = lambda t,P  : - self.f_vel(P)
        # self.f_odeint = lambda P,t  : - self.f_vel(P)
        self.f_ode = lambda t,P : - self.f_vel(P)/self.f_por(P)
        self.f_odeint = lambda P,t : - self.f_vel(P)/self.f_por(P)
        self.maxV = np.sqrt(np.max(self.Vx**2 + self.Vy**2))

        # Set also the "scipy.integrate.ode" integrator 
        self.ode = scipy.integrate.ode(self.f_ode)
        self.ode.set_integrator('dopri5')

    # Utility to deal with WELLs data
    def CoordDataWell(self, i=0) :
        """Auxiliary function that returns the coordinate of a well
        """
        idx = self.WELLs['idx'][i][1:]
        Xc = self.X[idx]
        Yc = self.Y[idx]
        R  = 0.5 * np.sqrt(self.deltaCOL[idx[0]]**2 + self.deltaROW[idx[1]]**2)
        return Xc, Yc, R

    def OriginFlowLinesWell(self, i=0, Norigin=40, radius=0.0) :
        """Auxiliary function used to generate the starting point of flows 
        that start form a given well. 
        """
        Xc, Yc, R = self.CoordDataWell(i)
        if radius !=0 :
            R = radius
        phi =  2 * np.pi * np.linspace(0.0, 1.0, Norigin) 
        X0  = Xc + R * np.cos(phi) 
        Y0  = Yc + R * np.sin(phi)
        return X0, Y0 

    #   Isochrone Utilities
    #  
    #   This utility compute the ISOCHRONE of a single or 
    #   multiple WELLs at a given time.
    #
    #   The additional function are just there to allow to implement 
    #   a refined strategy to deals with multiple stagnation points 
    #   in a controlled way
    #      
        
    #  Having points that are separated at most of "delta" time the length of the isochrone
    #  
    #  This is the main function that generate the isochrone for a single well.
    #  ComputeIsochroneWells call this function for any required well.  
    def ComputeIsochroneWells(self, delta=0.05, wells = [], time=60 ,radius=0.0, MAXSTEPS=10, DP=0.0) :
        """Generate isochrone for the time t for each given well using the main
        function ComputeIsochroneWellMAXdelta.

        Keyword arguments:

        delta --  initial fractional separation (with respect to the perimeter) of the 
                  initial point of the particles to be tracked (default 0.05)
        wells -- list of wells for which are computed the isochrones.
                 (default [] that correspond to all the wells) 
        time -- time of the isochrone in days (default 60 days) 
        radius -- radius around the well to be consider as the original time
                  (default 0, that correspond to use a circle around the well cell)
        MAXSTEPS -- maximum number of iterations (default 10)
        DP -- maximum distance of the final computed point in the isochrone
              (default 0, that correspond to use delta*perimeter of the isochrone)

        """
        if wells == [] :
            wells = range(self.WELLs['N'])
        return [self.ComputeIsochroneWellMAXdelta(delta, well, time, radius, MAXSTEPS, DP) for well in wells]
    
    # Main routine used to compute the ISOCHRONEs
    # use: self.ComputeIsochroneWell    (first GUESS)
    #      self.ComputeIsochroneORIGIN  (Generate new points)
    def ComputeIsochroneWellMAXdelta(self, delta=0.05, well=0, time=60, radius=0.0, MAXSTEPS=10, DP=0) :
        """Generate isochrone for the given time of the selected well

        Keyword arguments:

        delta --  initial fractional separation (with respect to the perimeter) of the 
                  initial point of the particles to be tracked (default 0.05)
        well -- index of the well to be considered for the computation of the isochrones
                (default 0) 
        time -- time of the isochrone in days (default 60 days) 
        radius -- radius around the well to be consider as the original time
                  (default 0, that correspond to use a circle around the well cell)
        MAXSTEPS -- maximum number of iterations (default 10)
        DP -- maximum distance of the final computed points in the isochrone
              (default 0, that correspond to use delta*perimeter of the isochrone)

        """
        if delta < 0.01 :
            delta = 0.01
        Xs, Ys, R, Xc, Yc, phi0s, phifs = self.ComputeIsochroneWell(well, time, Norigin = 1 + int(1/delta), radius=0.0)

        # If DP > 0 use this values for maximum distance
        # between points otherwise use percentual increments
        if DP > 0 :
            MAXdP = DP
        else :
            length = np.sum(np.sqrt(np.diff(Xs)**2 + np.diff(Ys)**2))
            MAXdP = length * delta

        # Start refining
        dP = np.sqrt(np.diff(Xs)**2 + np.diff(Ys)**2)
        REFINE = np.where(dP > MAXdP, 1, 0)
        nREFINE = int(np.sum(REFINE))

        # print nREFINE
        STEPS = 0
        while nREFINE > 1 and STEPS < MAXSTEPS :
            STEPS += STEPS
            NEWphi0s = np.zeros(len(phi0s) + nREFINE)
            i=0
            NEWphi0s[i] = phi0s[0] 
            # print len(REFINE),REFINE
            # print len(phi0s),phi0s
            for idx in range(1, len(phi0s)) :
                # print idx 
                if (REFINE[idx-1] == 1) :
                    if phi0s[idx] == 0.0 :
                        NEWphi0s[i + 1] = 0.5 * (phi0s[idx-1] + np.pi*2)
                        NEWphi0s[i + 2] = phi0s[idx]
                    else:  
                        NEWphi0s[i + 1] = 0.5 * (phi0s[idx-1] + phi0s[idx])
                        NEWphi0s[i + 2] = phi0s[idx]
                    i = i + 2
                else:
                    NEWphi0s[i + 1] = phi0s[idx]
                    i = i + 1
            # print len(NEWphi0s),NEWphi0s
            Xs, Ys, R, Xc, Yc, phi0s, phifs = self.ComputeIsochroneORIGIN(R, Xc, Yc, NEWphi0s, time) 
            dP = np.sqrt(np.diff(Xs)**2 + np.diff(Ys)**2)
            REFINE  = np.where(dP > MAXdP, 1, 0)
            nREFINE = int(np.sum(REFINE))
            # print nREFINE
        return Xs, Ys, time, well  #, R, Xc, Yc, phi0s, phifs 

    #  Base routine that would allow to do refinements
    def ComputeIsochroneORIGIN(self, R, Xc, Yc, phi0s, time) :  
        """Auxiliary function that generate isochrone for the time t starting
        from points centered in Xc, Yc and on a radius R at the angles phis
        """ 
        X0s = Xc + R * np.cos(phi0s) 
        Y0s = Yc + R * np.sin(phi0s)
        Nelem = len(X0s)
        phifs = np.zeros(Nelem)
        Xs = np.zeros(Nelem) 
        Ys = np.zeros(Nelem) 
        for idx in range(Nelem) :
            X0, Y0 = X0s[idx], Y0s[idx]
            self.ode.set_initial_value((X0, Y0), 0)
            Xf, Yf =self.ode.integrate(time)
            phif = np.arctan2(Yf - Yc, Xf - Xc)
            if phif < 0 :
                phif += 2.0 * np.pi
            # save computed values
            Xs[idx], Ys[idx] = Xf, Yf 
            phifs[idx] = phif
        return Xs, Ys, R, Xc, Yc, phi0s, phifs 
    
    #  Base routine that generate the first guess around a well
    def ComputeIsochroneWell(self, well=0, time=60 ,Norigin=20, radius=0.0) :
        """Auxiliary function that generate isocrone fo the given time around a 
        a given well
 
       well -- index of the well to be considered for the computation of the isochrones
                (default 0) 
        time -- time of the isochrone in days (default 60 days) 
        Norigin -- number of tracking particle to be used (default 20)
        radius -- radius around the well to be consider as the original time
                  (default 0, that correspond to use a circle around th ewell cell)
        """

        Xc, Yc, R = self.CoordDataWell(i=well) 
        X0s, Y0s = self.OriginFlowLinesWell(well, Norigin=Norigin, radius=radius)
        R = np.sqrt((X0s[0]-Xc)**2 + (Y0s[0]-Yc)**2)
        Nelem = len(X0s)
        phi0s = np.zeros(Nelem)
        phifs = np.zeros(Nelem)
        Xs = np.zeros(Nelem) 
        Ys = np.zeros(Nelem) 
        for idx in range(Nelem) :
            X0, Y0 = X0s[idx], Y0s[idx]
            self.ode.set_initial_value((X0, Y0), 0)
            Xf, Yf =self.ode.integrate(time)
            phi0 = np.arctan2(Y0 - Yc, X0 - Xc)
            phif = np.arctan2(Yf - Yc, Xf - Xc)
            if phi0 < 0 :
                phi0 += 2.0 * np.pi
            if phif < 0 :
                phif += 2.0 * np.pi
            # save computed values 
            Xs[idx], Ys[idx] = Xf, Yf 
            phi0s[idx], phifs[idx] = phi0, phif
        return Xs, Ys, R, Xc, Yc, phi0s, phifs 

    #  Exporting routine for Isochrone
    def ExportIsochrones(self,ISOs, fname,Origin=[],Alpha=0.0, fmt ='%.18e') :
        """Export the list computed isochrone  ISOs to a text file of name fname

        Keyword arguments:

        Origin -- Origin of the coordinate (defalt [], will use the model one) 
        Alpha -- rotation angle of the model (defalt 0.0, will use the model one) 

        """
        if type(ISOs) == tuple:
            ISOs= [ISOs]

        if len(Origin) != 2 :
            M = self.Rot
            Origin = self.Origin
        else :
            alpha = Alpha/180.0*np.pi  # Convert the angle from degree to radian
            Origin = np.array(Origin)
            M = np.array([[np.cos(alpha), -np.sin(alpha)],
                          [np.sin(alpha), np.cos(alpha)]])

        niso = len(ISOs)
        npoints = np.zeros(niso, dtype='int')
        for idx in range(niso) :
            npoints[idx] = len(ISOs[idx][0])
        TOTpoints = np.cumsum(npoints)
        print(str(npoints) + " " + str(TOTpoints))
        datas = np.zeros( (TOTpoints[-1], 4) )
        Px = Origin[0] + M[0, 0] * ISOs[0][0] + M[0,1] * ISOs[0][1]
        Py = Origin[1] + M[1, 0] * ISOs[0][0] + M[1, 1] * ISOs[0][1]
        datas[:TOTpoints[0], 0] = Px
        datas[:TOTpoints[0], 1] = Py
        datas[:TOTpoints[0], 2] = ISOs[0][2]
        datas[:TOTpoints[0], 3] = ISOs[0][3] 
        for idx in range(1, niso) :
            Px = Origin[0] + M[0,0] * ISOs[idx][0] + M[0, 1] * ISOs[idx][1]
            Py = Origin[1] + M[1,0] * ISOs[idx][0] + M[1, 1] * ISOs[idx][1]
            datas[(TOTpoints[idx-1]):TOTpoints[idx], 0] = Px
            datas[(TOTpoints[idx-1]):TOTpoints[idx], 1] = Py
            datas[(TOTpoints[idx-1]):TOTpoints[idx], 2] = ISOs[idx][2]
            datas[(TOTpoints[idx-1]):TOTpoints[idx], 3] = ISOs[idx][3]
        np.savetxt(fname, datas, fmt = fmt)
    
    def ExportIsochronesShapes(self,ISOs, fname,Origin=[],Alpha=0.0) :
        """Export the list of computed isochrone ISOs to a shape file of name fname

        Keyword arguments:

        Origin -- Origin of the coordinate (defalt [], will use the model one) 
        Alpha -- rotation angle of the model (defalt 0.0, will use the model one) 

        """

        if ('shapefile' in sys.modules) == False :
            print("Can't produce shapefile. Please install shapefile")
            return

        if len(Origin) != 2 :
            M = self.Rot
            Origin = self.Origin
        else :
            alpha = Alpha/180.0*np.pi  # Convert the angle from degree to radian
            Origin = np.array(Origin)
            M = np.array([[np.cos(alpha), -np.sin(alpha)],
                          [np.sin(alpha), np.cos(alpha)]])

        niso = len(ISOs)
        w = shapefile.Writer()
        w.field('TIME', 'C', '8')
        w.field('AREA', 'C', '4')
        w.field('DESCRIPTION', 'C', '68')
        for idx in range(0, niso) :
            Px = Origin[0] + M[0,0] * ISOs[idx][0] + M[0, 1] * ISOs[idx][1]
            Py = Origin[1] + M[1,0] * ISOs[idx][0] + M[1, 1] * ISOs[idx][1]
            time = ISOs[idx][2]
            well = ISOs[idx][3]
            listP = [ [Px[i], Py[i]] for i in range(len(Px))]
            STRtime = ('%8.2f' % time)
            STRwell = ('%4d' % well)
            w.poly(parts=[listP], shapeType = shapefile.POLYLINE)
            w.record(STRtime,STRwell,'Isochrone at time ' + STRtime + ' (d) of capture area ' + STRwell)
        w.save(fname)


    # Visualitzation Utility 

    # Utility functions to plots auxiliary graph
    def CreateFigure(self) :
        """ Create a figure and axes with the correct aspect ratio of 
        the model and return the corresponding figure and axes opbjects
        """

        mSIZEx = self.bbox[1]-self.bbox[0]
        mSIZEy = self.bbox[3]-self.bbox[2]
        ASPECT_RATIO = mSIZEy/mSIZEx
        SIZEx = 6.0    
        SIZEy = (1.0/150) * np.int(SIZEx*ASPECT_RATIO*150)

        fig = plt.figure(figsize=(SIZEx, SIZEy))
        ax = fig.add_axes([0.16,0.18,0.80,0.80])
        ax.set_xlabel(r'$x \, (m)$')
        ax.set_ylabel(r'$y \, (m)$')
        self.fig = fig
        self.ax = ax
        return fig, ax

    def PlotStreamLines(self, ax='', density=0.3) :
        """ Plot automatic stream lines for the 2 dimensional flow of 
        SteadyFlow model

        Keyword arguments:

        density -- density of the flow line (default 0.3)
        ax -- axis to use to use (default the one associate with the flow)
 
        """

        if  (type(ax) != matplotlib.axes._axes.Axes ) :
            ax = self.ax
        ax.streamplot(self.X, self.Y, self.Vx, self.Vy, density=density)

    def PlotWells(self, ax = '') :
        """ Plot the position of the well on the SteadyFlow model

        Keyword arguments:

        ax -- axis to use to use (default the one associate with the flow)
 
       """
        if  (type(ax) != matplotlib.axes._axes.Axes ) :
            ax = self.ax
        for idx in range(self.WELLs['N']) :
            idx2D = self.WELLs['idx'][idx][1:3]
            X,Y = self.X[idx2D], self.Y[idx2D]
            print(("Wells %d  (%d,%d) at  X= %8.1f  Y=%8.1f" %(idx, idx2D[0], idx2D[1], X, Y)))
            ax.plot(X, Y, 'ro')
    
    # Compute FLOW LINEs using "scipy.integrate.odeint"
    def ComputeFlow(self, X0, Y0, t) :
        """Compute a flow (backward in time) starting from X0,Y0 for the times t
        """
        TRAJECTORY = scipy.integrate.odeint(self.f_odeint, [X0, Y0], t)
        Xt = TRAJECTORY[:, 0] 
        Yt = TRAJECTORY[:, 1]
        return Xt, Yt  

    def ComputeFlows(self, X0s, Y0s, t) :
        """Compute multip,e flow (backward in time) starting from a series of
        points X0s,Y0 for the times t
        """
        nX0s = len(X0s)
        nt = len(t)
        XFlux = np.zeros((nt, nX0s))
        yFlux = np.zeros((nt, nX0s))
        for i in range(nX0s) :
            Xt, Yt = self.ComputeFlow(X0s[i], Y0s[i], t) 
            XFlux[:, i] = Xt 
            yFlux[:, i] = Yt 
        return XFlux, yFlux

    # Function that print an usage example of the module
def Usage() :
    """## Usage example for a model of name "test" in current directory
    import Isochrone 
    m=Isochrone.SteadyFlow('test',LAYER=0)  
    m.ComputeVelocityFieldInterpolator(porosity=0.3)    # constant porosity 
    fig,ax=m.CreateFigure()
    m.PlotStreamLines() 
    m.PlotWells()
    times = [60,365,3*365]
    linec = {60:'r',365:'g',3*365:'m'}
    lines = dict()
    Niso= len(times)
    wells = [] 
    ISOs   = [] 
    for i in range(Niso):
        ISO  = m.ComputeIsochroneWells(time=times[i],wells=wells,MAXSTEPS=60,DP=5)
        ISOs = ISOs + ISO
    for i in range(len(ISOs)) :
        PX , PY = ISOs[i][:2]
        color = linec[ ISOs[i][2] ]
        lines[i],=ax.plot(PX,PY,color,lw=2)
    fig.savefig('test.pdf')
    
    m.ExportIsochrones(ISOs,"test_iso.txt")
    m.ExportIsochronesShapes(ISOs,"test_iso_shape")
    """
    print("Usage: # import Isochrone")

#  From command line
if __name__ == "__main__":

    import sys
    import os
    import getopt

    # PARSE command line arguments
    HELP_LINE = """./Isochrone.py ... 
    (mandatory argument: -i plus at least one of -p or -t)
    
    -i <Model>    Model Name  
    -b <BaseDir>  Base dir where to find the model
    -o <output>   Base name for output
    -P            Produce a plot
    -t            Produce the output .txt file
    
    -w "0,1,2"    List of wells to be considered
    -l <layer>    Layer to be considered
    -p <porosity> Porosity of the medium (a vaule or a filename)
    
    -x <X>        X coordinate of the upper left corner
    -y <Y>        Y coordinate of the upper left corner
    -a <angle>    Rotation angle in degree
    
    list of times (for example: 60 365 ... (in days)) 
"""

    ModelName = ''
    OutName = ''
    BaseDir = '.'
    UseParFile = False
    ProducePlot = False
    ProduceTXT = False
    originX = 0.0
    originY = 0.0
    alpha = 0.0
    porosity = 1.0
    wells = []
    LAYER = 0

    try :

        opts, args = getopt.getopt(sys.argv[1:],"hPtl:p:w:b:i:o:x:y:a:f:",["ifile=","ofile="])

        # END PARSE command line arguments
        for opt, arg in opts:
            if (opt == '-i') :
                ModelName = arg

            elif (opt == '-P') :
                ProducePlot = True

            elif (opt == '-t') :
                ProduceTXT  = True

            elif (opt == '-f') :
                UseParFile  = True
                ParFile     = arg

            elif (opt == '-o') :
                OutName = arg

            elif (opt == '-b') :
                BaseDir = arg

            elif (opt == '-l') :
                LAYER = int(arg)

            elif (opt == '-p') :
                if (os.path.exists(arg) == True) :
                    porosity = arg
                else:
                    porosity = float(arg)
            elif (opt == '-w') :
                wells = eval(arg)

            elif (opt == '-x') : 
                originX = float(arg)

            elif (opt == '-y') : 
                originY = float(arg)

            elif (opt == '-a') :
                alpha = float(arg)

            elif (opt == '-h') :
                print(HELP_LINE)

        if (UseParFile  == True) :
            pass;
            # Here we set the arguments from the ParFile

        if (len(OutName) < 1) :    
            OutName = "OUT_" + ModelName

        if (len(args) >= 1 and len(ModelName) and (ProducePlot or ProduceTXT)) :
            times  = np.array([float(x) for x in args])
            print( " -------------------------------------------------- ")
            print( "  Selected execution with the following parameters"  )
            print( " -------------------------------------------------- ")
            print( "ProducePlot "  + str( ProducePlot )                  )
            print( "ProduceTXT  "  + str( ProduceTXT  )                  )
            print( "ModelName   "  + str( ModelName   )                  )
            print( "BaseDir     "  + str( BaseDir     )                  )
            print( "OutName     "  + str( OutName     )                  )
            print( "porosity    "  + str( porosity    )                  )
            print( "wells       "  + str( wells       )                  )
            print( "LAYER       "  + str( LAYER       )                  )
            print( "Times       "  + str( times       )                  )
            print( " -------------------------------------------------- ")

            FullName = os.path.join(BaseDir,ModelName)
            m=SteadyFlow(FullName,LAYER=LAYER,UPPERLEFTx=originX,UPPERLEFTy=originY,Alpha=alpha)
            m.ComputeVelocityFieldInterpolator(porosity=porosity)
            Niso= len(times)
            ISOs   = [] ## -----

            for i in range(Niso):
                ISO  = m.ComputeIsochroneWells(time=times[i],wells=wells,MAXSTEPS=60,DP=5)
                ISOs = ISOs + ISO

            if ProducePlot :
                # Plot matter
                fig,ax = m.CreateFigure()
                m.PlotWells(ax)
                lines = {}
                for idx in range(len(ISOs)) : 
                    PX , PY = ISOs[idx][:2]
                    lines[i],=ax.plot(PX,PY,'k',lw=2)
                fig.savefig(OutName+'.pdf')

            if ProduceTXT :
                # Export matter
                m.ExportIsochrones(ISOs,OutName+".txt")
                m.ExportIsochronesShapes(ISOs,OutName+".txt")
    except :
        pass;
        print("Command line execution generated an exception. Please check arguments")
#
