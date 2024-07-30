
module deb_module

  use, intrinsic :: iso_c_binding

  implicit none

  interface

    subroutine deb_fct_c(n, Tin, Tout) bind(C, name = "deb_fct_c")

      import c_int, c_double
      integer(c_int), intent(in), value :: n
      real(c_double), intent(in)        :: Tin(n)
      real(c_double), intent(in)        :: Tout(n)

    end subroutine deb_fct_c

  end interface

end module deb_module


