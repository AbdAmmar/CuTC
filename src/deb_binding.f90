
program deb_binding

    use deb_module

    implicit none

    integer :: i, n
    double precision, allocatable :: Tin(:)
    double precision, allocatable :: Tout(:)

    n = 10
    allocate(Tin(n))
    allocate(Tout(n))

    Tin = 1.d0

    print *, " start call to deb_fct_c"

    call deb_fct_c(n, Tin, Tout)

    print *, " end deb_fct_c"

    deallocate(Tin)

    print *, "Tout:"
    do i = 1, n
      print *, i, Tout(i)
    enddo
    deallocate(Tout)

end program deb_binding


