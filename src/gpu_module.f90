
module gpu_module

  use, intrinsic :: iso_c_binding

  implicit none

  interface

    subroutine tc_int_c(nBlocks, blockSize,                     &
                        n_grid1, n_grid2, n_ao, n_nuc, size_bh, &
                        r1, wr1, r2,  wr2, rn,                  &
                        aos_data1, aos_data2,                   &
                        c_bh, m_bh, n_bh, o_bh,                 &
                        int2_grad1_u12_ao, int_2e_ao) bind(C, name = "tc_int_c")

      import c_int, c_double
      integer(c_int), intent(in), value :: nBlocks, blockSize
      integer(c_int), intent(in), value :: n_grid1, n_grid2
      integer(c_int), intent(in), value :: n_ao
      integer(c_int), intent(in), value :: n_nuc
      integer(c_int), intent(in), value :: size_bh
      real(c_double), dimension(n_grid1,3), intent(in) :: r1
      real(c_double), dimension(n_grid1), intent(in) :: wr1
      real(c_double), dimension(n_grid2,3), intent(in) :: r2
      real(c_double), dimension(n_grid2), intent(in) :: wr2
      real(c_double), dimension(n_nuc,3), intent(in) :: rn
      real(c_double), dimension(n_grid1,n_ao,4), intent(in) :: aos_data1
      real(c_double), dimension(n_grid2,n_ao,4), intent(in) :: aos_data2
      real(c_double), dimension(size_bh,n_nuc), intent(in) :: c_bh
      integer(c_int), dimension(size_bh,n_nuc), intent(in) :: m_bh
      integer(c_int), dimension(size_bh,n_nuc), intent(in) :: n_bh
      integer(c_int), dimension(size_bh,n_nuc), intent(in) :: o_bh
      real(c_double), dimension(n_ao,n_ao,n_grid1,4), intent(out) :: int2_grad1_u12_ao
      real(c_double), dimension(n_ao,n_ao,n_ao,n_ao), intent(out) :: int_2e_ao

    end subroutine tc_int_c

  end interface

end module gpu_module


