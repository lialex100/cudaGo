package main

import (
	"math/rand"
	"reflect"
	"syscall"
	"time"
	"unsafe"
)

const N = 1 << 20

type Lib struct {
	dll        *syscall.DLL
	deinitProc *syscall.Proc
	dotProc    *syscall.Proc
	handler    uintptr
	X, Y       []float32
}

func LoadLib() (*Lib, error) {
	l := &Lib{}
	var err error
	defer func() {
		if nil != err {
			l.Release()
		}
	}()

	if l.dll, err = syscall.LoadDLL("cuda.dll"); nil != err {
		return nil, err
	}
	if l.deinitProc, err = l.dll.FindProc("deinit"); nil != err {
		return nil, err
	}
	if l.dotProc, err = l.dll.FindProc("dot"); nil != err {
		return nil, err
	}

	proc, err := l.dll.FindProc("init")
	if nil != err {
		return nil, err
	}
	proc.Call(uintptr(unsafe.Pointer(&l.handler)), uintptr(N))

	proc, err = l.dll.FindProc("getInputs")
	if nil != err {
		return nil, err
	}
	xh := (*reflect.SliceHeader)(unsafe.Pointer(&l.X))
	yh := (*reflect.SliceHeader)(unsafe.Pointer(&l.Y))
	xh.Len, xh.Cap, yh.Len, yh.Cap = N, N, N, N
	proc.Call(l.handler,
		uintptr(unsafe.Pointer(&xh.Data)), uintptr(unsafe.Pointer(&yh.Data)))

	return l, nil
}

func (l *Lib) Release() {
	if nil != l.deinitProc && 0 != l.handler {
		l.deinitProc.Call(l.handler)
	}
	if nil != l.dll {
		l.dll.Release()
	}
}

func (l *Lib) Dot() float32 {
	var r float32
	l.dotProc.Call(l.handler, uintptr(unsafe.Pointer(&r)))
	return r
}

func main() {
	lib, err := LoadLib()
	if nil != err {
		println(err.Error())
		return
	}
	defer lib.Release()

	rand.Seed(time.Now().Unix())
	x, y := lib.X, lib.Y
	for i := 0; i < N; i++ {
		x[i], y[i] = rand.Float32(), rand.Float32()
	}

	t := time.Now()
	var r float32
	for i := 0; i < 10000; i++ {
		r = 0
		for i := 0; i < N; i++ {
			r += x[i] * y[i]
		}
	}
	println("--------------------------------")
	println("CPU")
	println(time.Now().Sub(t).Microseconds())
	println(r)

	println("--------------------------------")
	println("GPU")
	t = time.Now()
	for i := 0; i < 10000; i++ {
		r = lib.Dot()
	}
	println(time.Now().Sub(t).Microseconds())
	println(r)
}
