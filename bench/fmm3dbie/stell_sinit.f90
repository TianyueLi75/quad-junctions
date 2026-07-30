! ---------------------------------------------------------------------------
!  Validation of the S_init harness against Table 1 of Greengard, O'Neil, Rachh et al. 2021:
!  reproduce the paper's OWN stellarator geometry (fmm3dbie get_stellarator_npat, the exact Eq-6.1
!  surface) at N_patches=2400, p=8, and sweep eps -- should match Table 1a (alpha), 1b (m),
!  1c (S_init) for the p=8 row.  Same near-quad timing path as s_init_sweep.f90.
!
!  Usage:  ./stell_sinit [norder] [nu] [nv]     (default: 7  20  60  ->  p=8, N_patches=2400)
! ---------------------------------------------------------------------------
program stell_sinit
  implicit none

  integer *8 :: nuv(2), norder, iptype0, npatches, npts, npols
  integer *8, allocatable :: norders(:), ixyzs(:), iptype(:)
  real *8, allocatable :: srcvals(:,:), srccoefs(:,:), qwts(:)

  integer *8, allocatable :: row_ptr(:), col_ind(:), iquad(:)
  integer *8 :: nnz, nquad, ndtarg, iptype_avg, norder_avg
  real *8, allocatable :: cms(:,:), rads(:), rad_near(:), targs(:,:)
  real *8 :: rfac, rfac0

  complex *16, allocatable :: wnear(:), wnear2(:)
  real *8 :: t_slp, t_dlp, tquadgen
  complex *16 :: zpars(3)
  integer *8 :: iquadtype

  integer *8, allocatable :: novers(:), ixyzso(:)
  integer *8 :: npts_over, ikerorder

  integer *8 :: i, ie
  real *8 :: t1, t2, omp_get_wtime, tinit, sinit, area, alpha, mmem
  real *8 :: clos(3), eps, epslist(4)
  character(len=64) :: arg

  norder = 7; nuv(1) = 20; nuv(2) = 60      ! p=8, N_patches = 2*20*60 = 2400
  call get_command_argument(1, arg); if (len_trim(arg).gt.0) read(arg,*) norder
  call get_command_argument(2, arg); if (len_trim(arg).gt.0) read(arg,*) nuv(1)
  call get_command_argument(3, arg); if (len_trim(arg).gt.0) read(arg,*) nuv(2)
  iptype0 = 1                                ! RV triangles (quads split by a diagonal), as in the paper

  call get_stellarator_npat_mem(nuv, norder, iptype0, npatches, npts)
  npols = (norder+1)*(norder+2)/2
  print '(a,i0,a,i0,a,i0,a,i0,a,i0,a,i0)', ' paper stellarator: nuv=(', nuv(1), ',', nuv(2), &
        ')  norder=', norder, ' (p=', norder+1, ')  npatches=', npatches, '  npts=', npts

  allocate(norders(npatches), ixyzs(npatches+1), iptype(npatches))
  allocate(srccoefs(9,npts), srcvals(12,npts))
  call get_stellarator_npat(nuv, norder, iptype0, npatches, npts, &
       norders, ixyzs, iptype, srccoefs, srcvals)

  allocate(qwts(npts))
  call get_qwts(npatches, norders, ixyzs, iptype, npts, srcvals, qwts)
  area = 0.0d0; clos = 0.0d0
  do i = 1, npts
     area = area + qwts(i)
     clos(1) = clos(1) + srcvals(10,i)*qwts(i)
     clos(2) = clos(2) + srcvals(11,i)*qwts(i)
     clos(3) = clos(3) + srcvals(12,i)*qwts(i)
  enddo
  print '(a,e14.7,a,3e11.3)', ' surface area =', area, '   closure int n dA =', clos

  ndtarg = 3
  allocate(targs(ndtarg,npts))
  do i = 1, npts
     targs(1,i) = srcvals(1,i); targs(2,i) = srcvals(2,i); targs(3,i) = srcvals(3,i)
  enddo
  iptype_avg = iptype(1); norder_avg = norder
  call get_rfacs(norder_avg, iptype_avg, rfac, rfac0)
  allocate(cms(3,npatches), rads(npatches), rad_near(npatches))
  call get_centroid_rads(npatches, norders, ixyzs, iptype, npts, srccoefs, cms, rads)
  do i = 1, npatches
     rad_near(i) = rads(i)*rfac
  enddo
  call findnearmem(cms, npatches, rad_near, ndtarg, targs, npts, nnz)
  allocate(row_ptr(npts+1), col_ind(nnz))
  call findnear(cms, npatches, rad_near, ndtarg, targs, npts, row_ptr, col_ind)
  allocate(iquad(nnz+1))
  call get_iquad_rsc(npatches, ixyzs, npts, nnz, row_ptr, col_ind, iquad)
  nquad = iquad(nnz+1) - 1

  zpars(1) = (1.0d0, 0.0d0)   ! zk = 1 (paper uses k=1 single layer)
  zpars(2) = (1.0d0, 0.0d0)
  zpars(3) = (0.0d0, 0.0d0)
  iquadtype = 1

  epslist = (/ 5.0d-3, 5.0d-4, 5.0d-7, 5.0d-10 /)
  print *, ''
  print *, ' === paper stellarator, Table 1 check: p=8 (norder=7), SLP Helmholtz k=1 ==='
  print *, '  p    eps         N        t_slp(s)   t_dlp(s)   tquadgen   S_slp(pts/s)  S_both(pts/s) alpha   m'
  do ie = 1, 4
     eps = epslist(ie)
     ikerorder = -1
     if (abs(zpars(3)) .gt. 1.0d-16) ikerorder = 0
     allocate(novers(npatches), ixyzso(npatches+1))
     call get_far_order(eps, npatches, norders, ixyzs, iptype, cms, rads, npts, &
          srccoefs, ndtarg, npts, targs, ikerorder, zpars(1), nnz, row_ptr, col_ind, &
          rfac, novers, ixyzso)
     npts_over = ixyzso(npatches+1) - 1
     alpha = dble(npts_over)/dble(npts)
     allocate(wnear(nquad), wnear2(nquad))
     wnear = 0.0d0; wnear2 = 0.0d0
     ! --- SLP near-quad, zpars=(zk,1,0) ---
     zpars(2) = (1.0d0,0.0d0); zpars(3) = (0.0d0,0.0d0)
     t1 = omp_get_wtime()
     call getnearquad_helm_comb_dir(npatches, norders, ixyzs, iptype, npts, &
          srccoefs, srcvals, eps, zpars, iquadtype, nnz, row_ptr, col_ind, &
          iquad, rfac0, nquad, wnear)
     t2 = omp_get_wtime(); t_slp = t2 - t1
     ! --- DLP near-quad, zpars=(zk,0,1) --- (the paper's perf-test times this too: tquadgen = SLP+DLP)
     zpars(2) = (0.0d0,0.0d0); zpars(3) = (1.0d0,0.0d0)
     t1 = omp_get_wtime()
     call getnearquad_helm_comb_dir(npatches, norders, ixyzs, iptype, npts, &
          srccoefs, srcvals, eps, zpars, iquadtype, nnz, row_ptr, col_ind, &
          iquad, rfac0, nquad, wnear2)
     t2 = omp_get_wtime(); t_dlp = t2 - t1
     tquadgen = t_slp + t_dlp
     mmem  = dble(nquad)/dble(npts)
     write(*,'(i5,1x,e11.3,1x,i9,1x,3(1x,e11.4),2(1x,e13.5),1x,f7.2,1x,f7.1)') &
          int(norder+1,4), eps, npts, t_slp, t_dlp, tquadgen, &
          dble(npts)/t_slp, dble(npts)/tquadgen, alpha, mmem
     deallocate(wnear, wnear2, novers, ixyzso)
  enddo
end program stell_sinit
