/* Stephen J. Brooks, 2010.
 * Source: https://en.wikipedia.org/wiki/User:Stephen_J._Brooks/algebraics/src
 * Licensed by the author under CC BY 3.0 (see Commons: Algebraicszoom.png).
 *
 * This will not compile: lset.c, graphics.c, WINMAIN and Complex live in
 * Brooks' private framework. Use algebraics.py instead.
 */
#include <lset.c>
#include <rnd/frnd.c>

char nonconv; int fq[5001];

void findroots_inner(Complex *c,const unsigned o,LSet *pr)
{
	Complex r;
	if (o==1)
	{
		r=-c[0]/c[1];
		LSet_add(pr,&r);
		return;
	}
	int n; Complex f,d,p,or;
	r=frnd(2)-1+I*(frnd(2)-1);
int i=0,j=0;
	do
	{
if (j==500) {r=frnd(2)-1+I*(frnd(2)-1); j=0;} else j++;
if (i>=5000) {nonconv=1; break;}
i++;
		or=r; f=0; d=0; p=1;
		for (n=0;n<o;n++,p*=r)
		{
			f+=p*c[n];
			d+=p*c[n+1]*(n+1);
		}
		f+=p*c[o];
		r-=f/d;
	}
	while (modsquared(r-or)>1e-20);
fq[i]++;
	LSet_add(pr,&r);
	for (n=o;n>0;n--) c[n-1]+=r*c[n];
	for (n=0;n<o;n++) c[n]=c[n+1];
	findroots_inner(c,o-1,pr);
}

Complex *findroots(Complex *c,const unsigned o)
{
	LSet r=LSet(Complex);
	findroots_inner(c,o,&r);
	free(c);
	return r.a;
}

#include <graphics.c>
#include <rnd/eithertime.c>
#include <rnd/sq.c>
#include <rnd/Mini.c>

GLuint othertex(const unsigned sz)
{
	GLuint ret; glGenTextures(1,&ret);
	glBindTexture(GL_TEXTURE_2D,ret);
	glTexParameterf(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR_MIPMAP_LINEAR);
	glTexParameterf(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);
	int n,x,y,xs=sz,ys=sz;
	unsigned char *td=malloc(xs*ys*3); float f;
	for (y=ys-1;y>=0;y--) for (x=xs-1;x>=0;x--)
	{
		n=(y*xs+x)*3;
		f=sq((float)sz/2)/(1+sq((float)x-xs/2)+sq((float)y-ys/2));
		td[n]=td[n+1]=td[n+2]=Mini(0xFF,f);
	}
	gluBuild2DMipmaps(GL_TEXTURE_2D,3,xs,ys,GL_RGB,GL_UNSIGNED_BYTE,td);
	free(td);
	return ret;
}

void putblob(const float x,const float y,const float r)
{
	glTexCoord2f(1,1); glVertex2f(x+r*16,y+r*16);
	glTexCoord2f(1,0); glVertex2f(x+r*16,y-r*16);
	glTexCoord2f(0,0); glVertex2f(x-r*16,y-r*16);
	glTexCoord2f(0,1); glVertex2f(x-r*16,y+r*16);
}

typedef struct {double x,y; int h,o;} Point;

LSet precalc(const int maxh)
{
	LSet ret=LSet(Point); Point p;
	int h,i,j,k,nz,l,sp;
for (i=0;i<=5000;i++) fq[i]=0;
	int temps=0,eqns=0,roots=0;
	for (h=2;h<=maxh;h++)
	{
		p.h=h;
		int *t=malloc(h*sizeof(int));
		for (i=(1<<(h-1))-1;i>=0;i-=2)
		{
			t[0]=0;
			for (j=h-2,k=0;j>=0;j--)
				if ((i>>j)&1) t[k]++; else {k++; t[k]=0;}
			temps++;
			if (k==0) continue;
			p.o=k;
			nz=0;
			for (j=k;j>=0;j--) if (t[j]!=0) nz++;
			for (j=(1<<(nz-1))-1;j>=0;j--)
			{
				Complex *c=malloc((k+1)*sizeof(Complex));
				for (l=k,sp=1;l>=0;l--)
					if (t[l]==0 || l==k) c[l]=t[l];
					else {c[l]=(j&sp?t[l]:-t[l]); sp<<=1;}
				eqns++;
					nonconv=0;
Complex *cc=malloc((k+1)*sizeof(Complex)); memcpy(cc,c,(k+1)*sizeof(Complex));
				c=findroots(c,k);
					if (!nonconv)
				for (l=k-1;l>=0;l--)
				{
					roots++;
					p.x=c[l].re; p.y=c[l].im;
					LSet_add(&ret,&p);
				}
					else
				{
					FILE *out=fopen("nonconv.log","at");
					for (l=k;l>=0;l--) fprintf(out,"%+lg*z^%d%s",cc[l].re,l,(l?"":"\n"));
					fclose(out);
				}
				free(c);
free(cc);
			}
		}
		free(t);
	}
	FILE *out=fopen("stats.txt","at");
	fprintf(out,"temps=%d eqns=%d roots=%d\n",temps,eqns,roots);
	fclose(out);
	return ret;
}

WINMAIN
{
	int n; gl_ortho=1;
	GRAPHICS(0,0,"Algebraic numbers [Stephen Brooks 2010]");
	GLuint tex=othertex(256),list=0;
	double ox=0,oy=0,zoom=yres/5,k1=0.125,k2=0.5;
	SetCursorPos(xres/2,yres/2);
	double ot=eithertime();
	LSet ps=precalc(15);
	LOOP
	{
		double dt=eithertime()-ot; ot=eithertime();
		ox+=(mx-xres/2)/zoom; oy+=(my-yres/2)/zoom;
		if (KEY(VK_O)) ox=oy=0;
		SetCursorPos(xres/2,yres/2);
		if (mb&1) zoom*=exp(dt*3); if (mb&2) zoom*=exp(-dt*3);
		if (KHIT(VK_Z)) {k1*=1.3; glDeleteLists(list,1); list=0;}
		if (KHIT(VK_X)) {k1/=1.3; glDeleteLists(list,1); list=0;}
		if (KHIT(VK_C)) {k2+=0.05; glDeleteLists(list,1); list=0;}
		if (KHIT(VK_V)) {k2-=0.05; glDeleteLists(list,1); list=0;}
		glMatrixMode(GL_MODELVIEW);
		glPushMatrix();
		glScaled(zoom,zoom,zoom);
		glTranslated((xres/2/zoom)-ox,(yres/2/zoom)-oy,0);
		if (!list)
		{
			list=glGenLists(1); glNewList(list,GL_COMPILE_AND_EXECUTE);
		glEnable(GL_BLEND);
		glBlendFunc(GL_ONE,GL_ONE);
		glDisable(GL_DEPTH_TEST);
		glEnable(GL_TEXTURE_2D);
		glBindTexture(GL_TEXTURE_2D,tex);
		glBegin(GL_QUADS);
		Point *p=ps.a;
		for (n=ps.m-1;n>=0;n--)
		{
			switch (p[n].o)
			{
				case 1: glColor3f(1,0,0); break;
				case 2: glColor3f(0,1,0); break;
				case 3: glColor3f(0,0,1); break;
				case 4: glColor3f(0.7,0.7,0); break;
				case 5: glColor3f(1,0.6,0); break;
				case 6: glColor3f(0,1,1); break;
				case 7: glColor3f(1,0,1); break;
				case 8: glColor3f(0.6,0.6,0.6); break;
				default: glColor3f(1,1,1); break;
			}
			putblob(p[n].x,p[n].y,k1*pow(k2,p[n].h-3));
		}
		glEnd();
			ot=eithertime();
			glEndList();
		}
		else if (list) glCallList(list);
		if (KEY(VK_L)) {glDeleteLists(list,1); list=0;}
		if (KEY(VK_CONTROL) && KHIT(VK_S)) screenshotauto();
		glMatrixMode(GL_MODELVIEW);
		glPopMatrix();
		ccl();
	}
}
