
module gpu_module

  use, intrinsic :: iso_c_binding

  implicit none

  interface

    subroutine tc_int_c(nBlocks, blockSize,                     &
                        n_grid1, n_grid2, n_ao, n_nuc, size_bh, &
                        r1, wr1, r2, wr2, rn,                   &
                        aos_data1, aos_data2,                   &
                        c_bh, m_bh, n_bh, o_bh,                 &
                        int2_grad1_u12_ao, int_2e_ao) bind(C, name = "tc_int_c")

      import c_int, c_double
      integer(c_int), intent(in), value :: nBlocks, blockSize
      integer(c_int), intent(in), value :: n_grid1, n_grid2
      integer(c_int), intent(in), value :: n_ao
      integer(c_int), intent(in), value :: n_nuc
      integer(c_int), intent(in), value :: size_bh
      real(c_double), intent(in)        :: r1(n_grid1,3)
      real(c_double), intent(in)        :: wr1(n_grid1)
      real(c_double), intent(in)        :: r2(n_grid2,3)
      real(c_double), intent(in)        :: wr2(n_grid2)
      real(c_double), intent(in)        :: rn(n_nuc,3)
      real(c_double), intent(in)        :: aos_data1(n_grid1,n_ao,4)
      real(c_double), intent(in)        :: aos_data2(n_grid2,n_ao,4)
      real(c_double), intent(in)        :: c_bh(size_bh,n_nuc)
      integer(c_int), intent(in)        :: m_bh(size_bh,n_nuc)
      integer(c_int), intent(in)        :: n_bh(size_bh,n_nuc)
      integer(c_int), intent(in)        :: o_bh(size_bh,n_nuc)
      real(c_double), intent(out)       :: int2_grad1_u12_ao(n_ao,n_ao,n_grid1,4)
      real(c_double), intent(out)       :: int_2e_ao(n_ao,n_ao,n_ao,n_ao)

    end subroutine tc_int_c

    ! ---

    subroutine deb_int_long_range(nBlocks, blockSize,            &
                                  n_grid2, n_ao, wr2, aos_data2, &        
                                  int_fct_long_range) bind(C, name = "deb_int_long_range")

      import c_int, c_double
      integer(c_int), intent(in), value :: nBlocks, blockSize
      integer(c_int), intent(in), value :: n_grid2
      integer(c_int), intent(in), value :: n_ao
      real(c_double), intent(in)        :: wr2(n_grid2)
      real(c_double), intent(in)        :: aos_data2(n_grid2,n_ao,4)
      real(c_double), intent(out)       :: int_fct_long_range(n_grid2,n_ao,n_ao)

    end subroutine deb_int_long_range

    ! ---

  end interface

end module gpu_module


