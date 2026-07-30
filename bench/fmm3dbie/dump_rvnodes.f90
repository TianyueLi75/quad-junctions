! Dump the order-11 Vioreanu-Rokhlin nodes (paper p=12: n_p = 12*13/2 = 78) on the
! reference triangle, in fmm3dbie's get_vioreanu_nodes ORDER, so the C++ exporter samples
! the surface at exactly the nodes surf_vals_to_coefs expects.  Output: rvnodes_o11.txt
program dump_rvnodes
  implicit none
  integer *8 :: norder, npols, i
  real *8, allocatable :: uvs(:,:), wts(:)
  character(len=256) :: arg, fname

  norder = 11
  call get_command_argument(1, arg)
  if (len_trim(arg) .gt. 0) read(arg,*) norder
  npols = (norder+1)*(norder+2)/2

  allocate(uvs(2,npols), wts(npols))
  call get_vioreanu_nodes_wts(norder, npols, uvs, wts)

  write(fname,'(a,i0,a)') 'rvnodes_o', norder, '.txt'
  open(unit=20, file=trim(fname), status='replace')
  write(20,'(2(1x,i0))') norder, npols
  do i = 1, npols
     write(20,'(3(1x,e25.16))') uvs(1,i), uvs(2,i), wts(i)
  end do
  close(20)

  print '(a,i0,a,i0,a,a)', ' dump_rvnodes: norder=', norder, ' npols=', npols, &
        '  -> ', trim(fname)
end program dump_rvnodes
