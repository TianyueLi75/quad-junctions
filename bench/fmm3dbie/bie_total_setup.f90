! ---------------------------------------------------------------------------
!  TOTAL setup timing for fmm3dbie on the exported Y-bifurcation surface, for the Laplace OR Stokes
!  single-layer kernel. Phases (correspond 1:1 to the SCTL BoundaryIntegralOp profile):
!     near-list build (findnear*)          ~ SCTL BuildNearLst (part of SetupNear)
!     far/oversample (get_far_order,...)   ~ SCTL SetupFarField  (excluded from the setup speed)
!     getnearquad (self+near)              ~ SCTL SetupSingular + SetupNear (quad part)
!  Speed = Nnodes / (near-list + getnearquad).  Laplace uses getnearquad_lap_comb_dir (scalar wnear);
!  Stokes uses getnearquad_stok_comb_vel (wnear(6,:), the 6 symmetric Stokeslet components).
!
!  Usage:  ./bie_total_setup <laplace|stokes> <surface_file> [eps]     (default eps 1e-8)
! ---------------------------------------------------------------------------
program bie_total_setup
  implicit none

  integer *8 :: npatches, npts, norder, npols
  integer *8, allocatable :: norders(:), ixyzs(:), iptype(:)
  real *8, allocatable :: srcvals(:,:), srccoefs(:,:)

  integer *8, allocatable :: row_ptr(:), col_ind(:), iquad(:)
  integer *8 :: nnz, nquad, ndtarg, iptype_avg, norder_avg, iquadtype
  real *8, allocatable :: cms(:,:), rads(:), rad_near(:), targs(:,:)
  real *8 :: rfac, rfac0

  real *8, allocatable :: wnear(:), wnear6(:,:)
  real *8 :: dpars(2)
  complex *16 :: zk

  integer *8, allocatable :: novers(:), ixyzso(:)
  integer *8 :: npts_over, ikerorder
  real *8, allocatable :: srcover(:,:), wover(:)

  ! DL const-density identity accuracy check
  real *8, allocatable :: rv_uvs(:,:), uvs_targ(:,:), sig1(:), pot1(:), sig3(:,:), pot3(:,:)
  integer *8, allocatable :: ipatch_id(:)
  real *8 :: dlerr, dlmean
  integer *8 :: kk

  integer *8 :: i, j, ios, nd9
  real *8 :: t0, t1, omp_get_wtime, t_nearlist, t_far, t_nearquad, t_total, eps, alpha
  character(len=32)  :: kernel
  character(len=256) :: fname, arg

  call get_command_argument(1, kernel)
  if (len_trim(kernel) .eq. 0) kernel = 'laplace'
  call get_command_argument(2, fname)
  if (len_trim(fname) .eq. 0) fname = 'ybifurc_ord12.srcvals'
  eps = 1.0d-8
  call get_command_argument(3, arg); if (len_trim(arg).gt.0) read(arg,*) eps
  if (trim(kernel).ne.'laplace' .and. trim(kernel).ne.'stokes') then
     print *, "ERROR: kernel must be 'laplace' or 'stokes' (got '", trim(kernel), "')"; stop
  endif

  open(unit=10, file=trim(fname), status='old', iostat=ios)
  if (ios .ne. 0) then
     print *, 'ERROR: cannot open ', trim(fname); stop
  endif
  read(10,*) npatches, norder, npts
  npols = (norder+1)*(norder+2)/2
  print '(a,a,a,i0,a,i0,a,i0,a,i0)', ' kernel=', trim(kernel), '  surface: npatches=', npatches, &
        ' norder=', norder, ' npols=', npols, ' npts=', npts

  allocate(srcvals(12,npts), srccoefs(9,npts))
  do i = 1, npts
     read(10,*) (srcvals(j,i), j=1,12)
  enddo
  close(10)

  allocate(norders(npatches), ixyzs(npatches+1), iptype(npatches))
  do i = 1, npatches
     norders(i) = norder; iptype(i) = 1; ixyzs(i) = (i-1)*npols + 1
  enddo
  ixyzs(npatches+1) = npatches*npols + 1
  nd9 = 9
  call surf_vals_to_coefs(nd9, npatches, norders, ixyzs, iptype, npts, &
       srcvals(1:9,1:npts), srccoefs)

  ! ---- phase 1: full near list ----
  ndtarg = 3
  allocate(targs(ndtarg,npts))
  do i = 1, npts
     targs(1,i) = srcvals(1,i); targs(2,i) = srcvals(2,i); targs(3,i) = srcvals(3,i)
  enddo
  iptype_avg = iptype(1); norder_avg = norder
  call get_rfacs(norder_avg, iptype_avg, rfac, rfac0)
  allocate(cms(3,npatches), rads(npatches), rad_near(npatches))
  t0 = omp_get_wtime()
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
  t1 = omp_get_wtime(); t_nearlist = t1 - t0

  ! ---- phase 2: far / oversampling (excluded from setup speed) ----
  dpars(1) = 1.0d0; dpars(2) = 0.0d0          ! single layer: alpha=1, beta=0
  ikerorder = -1
  if (abs(dpars(2)) .gt. 1.0d-16) ikerorder = 0
  allocate(novers(npatches), ixyzso(npatches+1))
  zk = (0.0d0, 0.0d0)                          ! Laplace/Stokes -> zero frequency for far-order estimate
  t0 = omp_get_wtime()
  call get_far_order(eps, npatches, norders, ixyzs, iptype, cms, rads, npts, &
       srccoefs, ndtarg, npts, targs, ikerorder, zk, nnz, row_ptr, col_ind, &
       rfac, novers, ixyzso)
  npts_over = ixyzso(npatches+1) - 1
  allocate(srcover(12,npts_over), wover(npts_over))
  call oversample_geom(npatches, norders, ixyzs, iptype, npts, srccoefs, srcvals, &
       novers, ixyzso, npts_over, srcover)
  call get_qwts(npatches, novers, ixyzso, iptype, npts_over, srcover, wover)
  t1 = omp_get_wtime(); t_far = t1 - t0
  alpha = dble(npts_over)/dble(npts)

  ! ---- phase 3: near-field quadrature correction (self + near) ----
  iquadtype = 1
  t0 = omp_get_wtime()
  if (trim(kernel) .eq. 'laplace') then
     allocate(wnear(nquad)); wnear = 0.0d0
     call getnearquad_lap_comb_dir(npatches, norders, ixyzs, iptype, npts, &
          srccoefs, srcvals, eps, dpars, iquadtype, nnz, row_ptr, col_ind, &
          iquad, rfac0, nquad, wnear)
  else
     allocate(wnear6(6,nquad)); wnear6 = 0.0d0
     call getnearquad_stok_comb_vel(npatches, norders, ixyzs, iptype, npts, &
          srccoefs, srcvals, eps, dpars, iquadtype, nnz, row_ptr, col_ind, &
          iquad, rfac0, nquad, wnear6)
  endif
  t1 = omp_get_wtime(); t_nearquad = t1 - t0

  t_total = t_nearlist + t_far + t_nearquad

  print *, ''
  print '(a,a,a)', ' === ', trim(kernel), '-SL TOTAL SETUP (fmm3dbie, full self+near list) ==='
  print '(a,i0,a,e10.3,a,i0,a,f6.2)', '   p=', norder+1, '   eps=', eps, &
        '   nnz=', nnz, '   alpha=', alpha
  print '(a,i0,a,i0)', '   Nnodes=', npts, '   nquad=', nquad
  print '(a,e12.4)', '   near-list build (~BuildNearLst)   t=', t_nearlist
  print '(a,e12.4)', '   far/oversample  (~SetupFarField)  t=', t_far
  print '(a,e12.4)', '   getnearquad self+near (~SetupSingular+SetupNear) t=', t_nearquad
  print '(a,e12.4,a,e13.5)', '   TOTAL setup t=', t_total, '   setup speed(nodes/s)=', dble(npts)/t_total

  ! ---- DL const-density identity accuracy check (untimed): D[1] -> -1/2 on-surface ----
  ! Self-contained *_eval (near-quad + FMM internally) with dpars=(0,1); on-surface targets = the
  ! discretization nodes, so ipatch_id/uvs_targ are the RV node locations. Reuses targs(3,npts).
  allocate(rv_uvs(2,npols)); call get_vioreanu_nodes(norder, npols, rv_uvs)
  allocate(ipatch_id(npts), uvs_targ(2,npts))
  do i = 1, npts
     ipatch_id(i) = (i-1)/npols + 1
     uvs_targ(1,i) = rv_uvs(1, mod(i-1,npols)+1)
     uvs_targ(2,i) = rv_uvs(2, mod(i-1,npols)+1)
  enddo
  dpars(1) = 0.0d0; dpars(2) = 1.0d0            ! pure double layer
  dlerr = 0.0d0; dlmean = 0.0d0
  if (trim(kernel) .eq. 'laplace') then
     allocate(sig1(npts), pot1(npts)); sig1 = 1.0d0
     call lap_comb_dir_eval(npatches, norders, ixyzs, iptype, npts, srccoefs, srcvals, &
          ndtarg, npts, targs, ipatch_id, uvs_targ, eps, dpars, sig1, pot1)
     do i = 1, npts
        dlerr = max(dlerr, abs(pot1(i) + 0.5d0)); dlmean = dlmean + pot1(i)
     enddo
     dlmean = dlmean/npts
  else
     allocate(sig3(3,npts), pot3(3,npts)); sig3 = 1.0d0
     call stok_comb_vel_eval(npatches, norders, ixyzs, iptype, npts, srccoefs, srcvals, &
          ndtarg, npts, targs, ipatch_id, uvs_targ, eps, dpars, sig3, pot3)
     do i = 1, npts
        do kk = 1, 3
           dlerr = max(dlerr, abs(pot3(kk,i) + 0.5d0)); dlmean = dlmean + pot3(kk,i)
        enddo
     enddo
     dlmean = dlmean/(3*npts)
  endif
  print '(a,e12.5,a,f9.5,a)', '   [accuracy] DL const-density identity: max|D[1]+0.5| = ', dlerr, &
        '   (mean D[1] = ', dlmean, ', expect -0.5)'
end program bie_total_setup
