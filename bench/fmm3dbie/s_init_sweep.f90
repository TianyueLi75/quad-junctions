! ---------------------------------------------------------------------------
!  S_init sweep -- a p=12 row of Table 1c of Greengard, O'Neil, Rachh et al. 2021.
!
!  Reads a triangular-patch surface (iptype=1, order-11 RV nodes = paper p=12) exported
!  from the quad-junctions Y-bifurcation, and for each quadrature tolerance eps measures
!  T_init = wall time of the near-field quadrature-correction precompute
!  (getnearquad_helm_comb_dir, single-layer Helmholtz k=1), then reports
!      S_init = N / T_init            (points processed per second, Table 1c)
!      alpha  = N_over / N            (oversampling parameter, Table 1a)
!      m      = nquad / N             (near entries per point, ~ memory/point, Table 1b)
!
!  The setup sequence (targets = on-surface points, get_rfacs -> get_centroid_rads ->
!  findnearmem -> findnear -> get_iquad_rsc -> [timed] getnearquad) is lifted verbatim from
!  fmm3dbie's own lpcomp_helm_comb_dir_addsub (src/helm_wrappers/helm_comb_dir.f:1944-2049).
!
!  Usage:  ./s_init_sweep [surface_file]      (default: ybifurc_p12.srcvals)
! ---------------------------------------------------------------------------
program s_init_sweep
  implicit none

  ! ---- surface arrays ----
  integer *8 :: npatches, npts, norder, npols
  integer *8, allocatable :: norders(:), ixyzs(:), iptype(:)
  real *8, allocatable :: srcvals(:,:), srccoefs(:,:), qwts(:)

  ! ---- near-field structure (eps-independent) ----
  integer *8, allocatable :: row_ptr(:), col_ind(:), iquad(:)
  integer *8 :: nnz, nquad, ndtarg, iptype_avg, norder_avg
  real *8, allocatable :: cms(:,:), rads(:), rad_near(:), targs(:,:)
  real *8 :: rfac, rfac0

  ! ---- near quadrature (per eps) ----
  complex *16, allocatable :: wnear(:)
  complex *16 :: zpars(3)
  integer *8 :: iquadtype

  ! ---- far-field oversampling (for alpha; not timed) ----
  integer *8, allocatable :: novers(:), ixyzso(:)
  integer *8 :: npts_over, ikerorder

  ! ---- misc ----
  integer *8 :: i, j, ios, ie, nd9
  real *8 :: t1, t2, omp_get_wtime, tinit, sinit, area, alpha, mmem
  real *8 :: clos(3), eps, epslist(4)
  character(len=256) :: fname

  call get_command_argument(1, fname)
  if (len_trim(fname) .eq. 0) fname = 'ybifurc_p12.srcvals'

  open(unit=10, file=trim(fname), status='old', iostat=ios)
  if (ios .ne. 0) then
     print *, 'ERROR: cannot open ', trim(fname)
     stop
  endif
  read(10,*) npatches, norder, npts
  npols = (norder+1)*(norder+2)/2
  print '(a,i0,a,i0,a,i0,a,i0)', ' surface: npatches=', npatches, &
        '  norder=', norder, '  npols=', npols, '  npts=', npts
  if (npatches*npols .ne. npts) print *, ' *** WARNING: npts != npatches*npols ***'

  allocate(srcvals(12,npts), srccoefs(9,npts))
  do i = 1, npts
     read(10,*, iostat=ios) (srcvals(j,i), j=1,12)
     if (ios .ne. 0) then
        print *, ' ERROR reading srcvals at point ', i
        stop
     endif
  enddo
  close(10)

  allocate(norders(npatches), ixyzs(npatches+1), iptype(npatches))
  do i = 1, npatches
     norders(i) = norder
     iptype(i)  = 1
     ixyzs(i)   = (i-1)*npols + 1
  enddo
  ixyzs(npatches+1) = npatches*npols + 1

  ! srccoefs from srcvals(1:9) -- fmm3dbie pattern (surf_routs.f90:1371)
  nd9 = 9
  call surf_vals_to_coefs(nd9, npatches, norders, ixyzs, iptype, npts, &
       srcvals(1:9,1:npts), srccoefs)

  ! geometry sanity: surface area and closure |int n dA| (watertightness of the split mesh)
  allocate(qwts(npts))
  call get_qwts(npatches, norders, ixyzs, iptype, npts, srcvals, qwts)
  area = 0.0d0; clos = 0.0d0
  do i = 1, npts
     area    = area    + qwts(i)
     clos(1) = clos(1) + srcvals(10,i)*qwts(i)
     clos(2) = clos(2) + srcvals(11,i)*qwts(i)
     clos(3) = clos(3) + srcvals(12,i)*qwts(i)
  enddo
  print '(a,e14.7)',   ' surface area   =', area
  print '(a,3e12.4)',  ' closure int n dA =', clos(1), clos(2), clos(3)

  ! ---- eps-independent near list (targets = on-surface points) ----
  ndtarg = 3
  allocate(targs(ndtarg,npts))
  do i = 1, npts
     targs(1,i) = srcvals(1,i)
     targs(2,i) = srcvals(2,i)
     targs(3,i) = srcvals(3,i)
  enddo

  iptype_avg = iptype(1)
  norder_avg = norder
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
  print '(a,i0,a,i0)', ' near list: nnz=', nnz, '  nquad=', nquad

  ! representation: single-layer Helmholtz k=1  ->  u = alpha*S_k + beta*D_k, (alpha,beta)=(1,0)
  zpars(1) = (1.0d0, 0.0d0)   ! zk = 1
  zpars(2) = (1.0d0, 0.0d0)   ! alpha (single layer)
  zpars(3) = (0.0d0, 0.0d0)   ! beta  (no double layer)
  iquadtype = 1

  epslist = (/ 5.0d-3, 5.0d-4, 5.0d-7, 5.0d-10 /)

  print *, ''
  print *, ' === Table 1c row: p=12 (norder=11), single-layer Helmholtz k=1 ==='
  print *, '    p        eps          N       T_init(s)      S_init(pts/s)    alpha      m'
  do ie = 1, 4
     eps = epslist(ie)

     ! oversampling estimate -> alpha (reported, NOT part of T_init)
     ikerorder = -1
     if (abs(zpars(3)) .gt. 1.0d-16) ikerorder = 0
     allocate(novers(npatches), ixyzso(npatches+1))
     call get_far_order(eps, npatches, norders, ixyzs, iptype, cms, rads, npts, &
          srccoefs, ndtarg, npts, targs, ikerorder, zpars(1), nnz, row_ptr, col_ind, &
          rfac, novers, ixyzso)
     npts_over = ixyzso(npatches+1) - 1
     alpha = dble(npts_over)/dble(npts)

     ! near-field quadrature correction -- THIS is T_init
     allocate(wnear(nquad))
     do i = 1, nquad
        wnear(i) = 0.0d0
     enddo
     t1 = omp_get_wtime()
     call getnearquad_helm_comb_dir(npatches, norders, ixyzs, iptype, npts, &
          srccoefs, srcvals, eps, zpars, iquadtype, nnz, row_ptr, col_ind, &
          iquad, rfac0, nquad, wnear)
     t2 = omp_get_wtime()

     tinit = t2 - t1
     sinit = dble(npts)/tinit
     mmem  = dble(nquad)/dble(npts)
     write(*,'(i5,1x,e12.3,1x,i10,1x,e13.4,1x,e15.5,1x,f9.2,1x,f9.1)') &
          12, eps, npts, tinit, sinit, alpha, mmem

     deallocate(wnear, novers, ixyzso)
  enddo

end program s_init_sweep
