c
c  Laplace S_init timing on the PAPER'S OWN stellarator (2021 fmm3dbie code, int4).
c  Same geometry as aquad_helm_stell_perf_test.f (igeomtype=2, iasp=3, iref=2 ->
c  ipars=(20,60), npatches=2400, norder=7 => p=8), same near list (rfac=2.75,
c  rfac0=1.25), but the SINGLE-LAYER LAPLACE near-quad (getnearquad_lap_comb_dir,
c  dpars=(1,0)) instead of Helmholtz.  Sweeps eps over the requested tolerances and
c  reports S_slp = npts / t_slp for each, the Table-1c metric.
c
c  Run single-threaded:  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 taskset -c <c> ./aquad_lap_2021
c
      implicit real *8 (a-h,o-z)
      real *8, allocatable :: srcvals(:,:),srccoefs(:,:),targs(:,:)
      real *8, allocatable :: wts(:)
      real *8, allocatable :: cms(:,:),rads(:),rad_near(:)
      integer ipars(2)
      integer, allocatable :: row_ptr(:),col_ind(:)
      integer, allocatable :: iquad(:)
      real *8, allocatable :: slp_near(:)
      real *8 dpars(2)

      integer, allocatable :: norders(:),ixyzs(:),iptype(:)
      integer, allocatable :: ipatch_id(:)
      real *8, allocatable :: uvs_targ(:,:)
      character *100 fname

      real *8 epslist(5)

      call prini(6,13)

      done = 1
      pi = atan(done)*4

      igeomtype = 2
      iasp = 3
      iref = 2

c     iasp=3 (low aspect, as used for the paper-matching Helmholtz run)
      ipars(1) = 5
      ipars(2) = 15
      ipars(1) = ipars(1)*2**(iref)
      ipars(2) = ipars(2)*2**(iref)

      npatches = 2*ipars(1)*ipars(2)

      norder = 7
      npols = (norder+1)*(norder+2)/2

      npts = npatches*npols
      allocate(srcvals(12,npts),srccoefs(9,npts))
      allocate(targs(3,npts))
      ifplot = 0

      call setup_geom(igeomtype,norder,npatches,ipars,
     1       srcvals,srccoefs,ifplot,fname)

      allocate(norders(npatches),ixyzs(npatches+1),iptype(npatches))
      do i=1,npatches
        norders(i) = norder
        ixyzs(i) = 1 +(i-1)*npols
        iptype(i) = 1
      enddo
      ixyzs(npatches+1) = 1+npols*npatches

      allocate(wts(npts))
      call get_qwts(npatches,norders,ixyzs,iptype,npts,srcvals,wts)

      allocate(cms(3,npatches),rads(npatches),rad_near(npatches))
      call get_centroid_rads(npatches,norders,ixyzs,iptype,npts,
     1     srccoefs,cms,rads)

      ndtarg = 3
      do i=1,npts
        targs(1,i) = srcvals(1,i)
        targs(2,i) = srcvals(2,i)
        targs(3,i) = srcvals(3,i)
      enddo

      allocate(ipatch_id(npts),uvs_targ(2,npts))
      do i=1,npts
        ipatch_id(i) = -1
        uvs_targ(1,i) = 0
        uvs_targ(2,i) = 0
      enddo
      call get_patch_id_uvs(npatches,norders,ixyzs,iptype,npts,
     1         ipatch_id,uvs_targ)

c     near field list (eps-independent: rfac,rfac0 fixed as in the Helmholtz run)
      call get_rfacs(norder,iptype,rfac,rfac0)
      rfac = 2.75d0
      rfac0 = 1.25d0
      do i=1,npatches
        rad_near(i) = rads(i)*rfac
      enddo

      call findnearmem(cms,npatches,rad_near,ndtarg,targs,npts,nnz)
      allocate(row_ptr(npts+1),col_ind(nnz))
      call findnear(cms,npatches,rad_near,ndtarg,targs,npts,row_ptr,
     1        col_ind)
      allocate(iquad(nnz+1))
      call get_iquad_rsc(npatches,ixyzs,npts,nnz,row_ptr,col_ind,
     1         iquad)
      nquad = iquad(nnz+1)-1
      allocate(slp_near(nquad))

      iquadtype = 1
c     SLP Laplace: u = alpha*S + beta*D, dpars=(alpha,beta)=(1,0)
      dpars(1) = 1.0d0
      dpars(2) = 0.0d0

      epslist(1) = 1.0d-3
      epslist(2) = 1.0d-5
      epslist(3) = 1.0d-7
      epslist(4) = 1.0d-9
      epslist(5) = 1.0d-11

      call prinf('npatches=*',npatches,1)
      call prinf('npts=*',npts,1)
      write(*,'(a)') ' '
      write(*,'(a)') ' 2021 fmm3dbie LAPLACE SLP on paper stellarator'
      write(*,'(a)') '   p    eps          npts       t_slp(s)   '//
     1   'S_slp(pts/s)'
      do ie = 1,5
        eps = epslist(ie)
        do i=1,nquad
          slp_near(i) = 0
        enddo
        t1 = omp_get_wtime()
        call getnearquad_lap_comb_dir(npatches,norders,
     1     ixyzs,iptype,npts,srccoefs,srcvals,ndtarg,npts,targs,
     2     ipatch_id,uvs_targ,eps,dpars,iquadtype,nnz,row_ptr,col_ind,
     3     iquad,rfac0,nquad,slp_near)
        t2 = omp_get_wtime()
        t_slp = t2-t1
        sinit = dble(npts)/t_slp
        write(*,'(i5,1x,e11.4,1x,i9,1x,e13.5,1x,e13.5)')
     1     norder+1, eps, npts, t_slp, sinit
      enddo

      stop
      end
c
c
c
      subroutine setup_geom(igeomtype,norder,npatches,ipars,
     1    srcvals,srccoefs,ifplot,fname)
      implicit real *8 (a-h,o-z)
      integer igeomtype,norder,npatches,ipars(*),ifplot
      character (len=*) fname
      real *8 srcvals(12,*), srccoefs(9,*)
      real *8, allocatable :: uvs(:,:),umatr(:,:),vmatr(:,:),wts(:)

      real *8, pointer :: ptr1,ptr2,ptr3,ptr4
      integer, pointer :: iptr1,iptr2,iptr3,iptr4
      real *8, target :: p1(10),p2(10),p3(10),p4(10)
      real *8, allocatable, target :: triaskel(:,:,:)
      real *8, allocatable, target :: deltas(:,:)
      integer, allocatable :: isides(:)
      integer, target :: nmax,mmax

      procedure (), pointer :: xtri_geometry

      external xtri_stell_eval,xtri_sphere_eval

      npols = (norder+1)*(norder+2)/2
      allocate(uvs(2,npols),umatr(npols,npols),vmatr(npols,npols))
      allocate(wts(npols))

      call vioreanu_simplex_quad(norder,npols,uvs,umatr,vmatr,wts)

      if(igeomtype.eq.1) then
        itype = 2
        allocate(triaskel(3,3,npatches))
        allocate(isides(npatches))
        npmax = npatches
        ntri = 0
        call xtri_platonic(itype, ipars(1), npmax, ntri,
     1      triaskel, isides)

        xtri_geometry => xtri_sphere_eval
        ptr1 => triaskel(1,1,1)
        ptr2 => p2(1)
        ptr3 => p3(1)
        ptr4 => p4(1)

        if(ifplot.eq.1) then
           call xtri_vtk_surf(fname,npatches,xtri_geometry, ptr1,ptr2,
     1         ptr3,ptr4, norder,'Triangulated surface of the sphere')
        endif

        call getgeominfo(npatches,xtri_geometry,ptr1,ptr2,ptr3,ptr4,
     1     npols,uvs,umatr,srcvals,srccoefs)
      endif

      if(igeomtype.eq.2) then
        done = 1
        pi = atan(done)*4
        umin = 0
        umax = 2*pi
        vmin = 2*pi
        vmax = 0
        allocate(triaskel(3,3,npatches))
        nover = 0
        call xtri_rectmesh_ani(umin,umax,vmin,vmax,ipars(1),ipars(2),
     1     nover,npatches,npatches,triaskel)

        mmax = 2
        nmax = 1
        xtri_geometry => xtri_stell_eval

        allocate(deltas(-1:mmax,-1:nmax))
        deltas(-1,-1) = 0.17d0
        deltas(0,-1) = 0
        deltas(1,-1) = 0
        deltas(2,-1) = 0

        deltas(-1,0) = 0.11d0
        deltas(0,0) = 1
        deltas(1,0) = 4.5d0
        deltas(2,0) = -0.25d0

        deltas(-1,1) = 0
        deltas(0,1) = 0.07d0
        deltas(1,1) = 0
        deltas(2,1) = -0.45d0

        ptr1 => triaskel(1,1,1)
        ptr2 => deltas(-1,-1)
        iptr3 => mmax
        iptr4 => nmax

        if(ifplot.eq.1) then
           call xtri_vtk_surf(fname,npatches,xtri_geometry, ptr1,ptr2,
     1         iptr3,iptr4, norder,
     2         'Triangulated surface of the stellarator')
        endif

        call getgeominfo(npatches,xtri_geometry,ptr1,ptr2,iptr3,iptr4,
     1     npols,uvs,umatr,srcvals,srccoefs)
      endif

      return
      end
